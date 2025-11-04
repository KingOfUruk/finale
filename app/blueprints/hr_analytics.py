from flask import Blueprint, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.types import Integer, String, Numeric
from sqlalchemy.dialects.oracle import NUMBER
from datetime import datetime, timedelta
import warnings
import logging

from app.config.database import get_oracle_credentials

warnings.filterwarnings('ignore')

# Create HR analytics blueprint
hr_analytics_bp = Blueprint('hr_analytics', __name__, template_folder='templates')
CORS(hr_analytics_bp)

class AnalyseurPointageRH:
    def __init__(self, credentials=None):
        credentials = credentials or get_oracle_credentials()
        driver = credentials.get('driver', 'oracle+oracledb')
        port = str(credentials.get('port', '1521'))
        self.chaine_connexion_oracle = (
            f"{driver}://{credentials['username']}:{credentials['password']}"
            f"@{credentials['host']}:{port}/?service_name={credentials['service_name']}"
        )
        self.moteur = None
        self.df_temps = None
        self.df_employe = None
        self.df_pointage = None
        self.df_service = None
        self.df_categories = None
        self.df_fusionne = None
        self.connect()  # Initialize connection on instantiation

    def connect(self):
        """Établir la connexion à la base de données"""
        try:
            self.moteur = create_engine(self.chaine_connexion_oracle)
            with self.moteur.connect() as conn:
                conn.execute(text("SELECT 1 FROM dual"))
            logging.info("Connexion réussie à la base de données Oracle")
            return True
        except Exception as e:
            logging.error(f"Échec de la connexion : {e}")
            self.moteur = None
            return False

    def ensure_connection(self):
        """Ensure database connection is established"""
        if self.moteur is None:
            return self.connect()
        return True

    def charger_donnees(self):
        try:
            if not self.ensure_connection():
                logging.error("Impossible de charger les données : connexion à la base de données non établie")
                return False

            self.df_temps = pd.read_sql("SELECT * FROM DIM_TEMPS_nouveau", self.moteur)
            self.df_employe = pd.read_sql("SELECT * FROM DIM_EMPLOYEe_nouveau", self.moteur)
            self.df_pointage = pd.read_sql("SELECT * FROM FAIT_POINTAGE", self.moteur)
            self.df_service = pd.read_sql("SELECT * FROM DIM_SERVICE", self.moteur)
            
            # Load categories if available
            try:
                self.df_categories = pd.read_sql("SELECT cod_cat, lib_cat FROM DIM_EMPLOYEe_nouveau GROUP BY cod_cat, lib_cat", self.moteur)
            except:
                self.df_categories = pd.DataFrame({'cod_cat': [1,2,3,4], 'lib_cat': ['Direction','Principal','Execution','Autre']})
            
            for df in [self.df_temps, self.df_employe, self.df_pointage, self.df_service]:
                df.columns = df.columns.str.lower()
            
            if self.df_categories is not None:
                self.df_categories.columns = self.df_categories.columns.str.lower()
                
            return True
        except Exception as e:
            logging.error(f"Erreur lors du chargement des données : {e}")
            return False

    def nettoyer_et_preparer_donnees(self):
        try:
            self.df_pointage['date_point'] = pd.to_datetime(self.df_pointage['date_point'], 
                                                          format='%d-%b-%y', errors='coerce')
            self.df_pointage['duree_minutes'] = pd.to_numeric(self.df_pointage['duree_minutes'], errors='coerce').fillna(0)
            if 'nbr_pointages' in self.df_pointage.columns:
                self.df_pointage['nbr_pointages'] = pd.to_numeric(self.df_pointage['nbr_pointages'], errors='coerce').fillna(0)
            else:
                self.df_pointage['nbr_pointages'] = 0
            self.df_pointage.loc[self.df_pointage['duree_minutes'] < 0, 'duree_minutes'] = 0
            self.df_pointage['pointage_valide'] = self.df_pointage['duree_minutes'] > 0
            
            # Daily aggregation
            self.df_pointage_quotidien = self.df_pointage.groupby(['mat_pers', 'date_point']).agg({
                'duree_minutes': 'sum',
                'nbr_pointages': 'sum',
                'heure_entree': 'min',
                'heure_sortie': 'max',
                'est_present': 'first',
                'pointage_valide': 'max'
            }).reset_index()
            self.df_pointage_quotidien.rename(columns={'pointage_valide': 'a_duree_valide'}, inplace=True)
            self.df_pointage_quotidien['a_duree_valide'] = self.df_pointage_quotidien['a_duree_valide'].astype(bool)
            
            # Main merge
            self.df_fusionne = self.df_pointage_quotidien.merge(
                self.df_temps, left_on='date_point', right_on='date_jour', how='left'
            ).merge(
                self.df_employe, on='mat_pers', how='left'
            ).merge(
                self.df_service, left_on='code_serv', right_on='code_serv', how='left'
            )
            
            # Filter active employees
            self.df_fusionne = self.df_fusionne[~self.df_fusionne['statut'].isin(['Parti/Démissionné'])]
            self.df_fusionne['libelle_y'] = self.df_fusionne['libelle_y'].fillna('Service Inconnu')
            
            # Add year column for analysis
            self.df_fusionne['annee_pointage'] = self.df_fusionne['date_point'].dt.year
            self.df_fusionne['a_duree_valide'] = self.df_fusionne['a_duree_valide'].fillna(False)
            
            return True
        except Exception as e:
            logging.error(f"Erreur lors du nettoyage des données : {e}")
            return False

    def get_filtered_data(self, selected_year=None):
        """Return a filtered copy of the fused dataframe for the requested year."""
        if self.df_fusionne is None:
            return pd.DataFrame()
        df = self.df_fusionne.copy()
        if selected_year:
            return df[df['annee_pointage'] == selected_year].copy()
        return df

    def calculer_tendances_annuelles(self, selected_year=None):
        """Calculate yearly trends and drilldowns based on the selected year."""
        try:
            if self.df_fusionne is None or self.df_fusionne.empty:
                return {
                    'yearly_overview': [],
                    'monthly_current_year': [],
                    'service_yearly_performance': [],
                    'selected_year': selected_year
                }

            df_all = self.df_fusionne.copy()
            df_all = df_all.dropna(subset=['annee_pointage'])

            # Yearly summary statistics across all available years
            yearly_stats = df_all.groupby('annee_pointage').agg({
                'mat_pers': 'nunique',  # Unique employees
                'duree_minutes': ['sum', 'mean'],  # Total and average minutes
                'nbr_pointages': 'sum'  # Total pointages
            }).reset_index()
            
            # Flatten column names
            yearly_stats.columns = [
                'annee', 'employes_uniques', 'total_minutes',
                'duree_moyenne', 'total_pointages'
            ]
            yearly_stats['total_heures'] = (yearly_stats['total_minutes'] / 60).round(0).astype(int)
            yearly_stats['heures_moyennes_par_jour'] = (yearly_stats['duree_moyenne'] / 60).round(2)
            yearly_stats = yearly_stats.sort_values('annee')
            
            # Determine which year to drill into for monthly/service views
            available_years = yearly_stats['annee'].tolist()
            if selected_year and selected_year in available_years:
                drilldown_year = selected_year
            else:
                drilldown_year = available_years[-1] if available_years else None

            monthly_trends = []
            service_yearly = []

            if drilldown_year is not None:
                selected_df = df_all[df_all['annee_pointage'] == drilldown_year]

                monthly_trends_df = selected_df.groupby('mois').agg({
                    'mat_pers': 'nunique',
                    'duree_minutes': ['sum', 'mean']
                }).reset_index()
                monthly_trends_df.columns = ['mois', 'employes_actifs', 'total_minutes', 'duree_moyenne']
                monthly_trends_df['total_heures'] = (monthly_trends_df['total_minutes'] / 60).round(0).astype(int)
                monthly_trends = monthly_trends_df.to_dict('records')
                
                service_yearly_df = selected_df.groupby('libelle_y').agg({
                    'mat_pers': 'nunique',
                    'duree_minutes': 'sum'
                }).reset_index()
                service_yearly_df['annee_pointage'] = drilldown_year
                service_yearly_df['total_heures'] = (service_yearly_df['duree_minutes'] / 60).round(0).astype(int)
                service_yearly = service_yearly_df.to_dict('records')
            
            return {
                'yearly_overview': yearly_stats.to_dict('records'),
                'monthly_current_year': monthly_trends,
                'service_yearly_performance': service_yearly,
                'selected_year': drilldown_year
            }
        except Exception as e:
            logging.error(f"Erreur lors du calcul des tendances annuelles : {e}")
            return {
                'yearly_overview': [],
                'monthly_current_year': [],
                'service_yearly_performance': [],
                'selected_year': selected_year
            }

    def analyser_categories_employes(self, selected_year=None):
        """Analyze employee categories performance"""
        try:
            if self.df_categories is not None and len(self.df_categories) > 0:
                df_filtered = self.get_filtered_data(selected_year)
                if df_filtered.empty:
                    return []

                # Merge with categories
                df_with_cat = df_filtered.merge(
                    self.df_categories, left_on='cod_cat', right_on='cod_cat', how='left'
                )
                
                cat_analysis = df_with_cat.groupby('lib_cat').agg({
                    'mat_pers': 'nunique',
                    'duree_minutes': ['sum', 'mean'],
                    'nbr_pointages': 'sum'
                }).reset_index()
                
                cat_analysis.columns = ['categorie', 'nb_employes', 'total_minutes', 'moyenne_minutes', 'total_pointages']
                cat_analysis['heures_moyennes'] = (cat_analysis['moyenne_minutes'] / 60).round(2)
                cat_analysis['total_heures'] = (cat_analysis['total_minutes'] / 60).round(0).astype(int)
                
                # Remove rows with NaN categories
                cat_analysis = cat_analysis.dropna(subset=['categorie'])
                
                return cat_analysis.to_dict('records')
            else:
                return []
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse des catégories : {e}")
            return []

    def analyser_saisonnalite(self, selected_year=None):
        """Analyze seasonal patterns"""
        try:
            df_filtered = self.get_filtered_data(selected_year)
            if df_filtered.empty:
                return {
                    'seasonal_details': [],
                    'season_summary': []
                }

            seasonal_data = df_filtered.groupby(['saison', 'mois']).agg({
                'mat_pers': 'nunique',
                'duree_minutes': ['sum', 'mean']
            }).reset_index()
            
            seasonal_data.columns = ['saison', 'mois', 'employes_actifs', 'total_minutes', 'moyenne_minutes']
            seasonal_data['total_heures'] = (seasonal_data['total_minutes'] / 60).round(0).astype(int)
            seasonal_data['heures_moyennes'] = (seasonal_data['moyenne_minutes'] / 60).round(2)
            
            # Season summary
            season_summary = df_filtered.groupby('saison').agg({
                'mat_pers': 'nunique',
                'duree_minutes': ['sum', 'mean']
            }).reset_index()
            season_summary.columns = ['saison', 'employes_moyens', 'total_minutes', 'moyenne_minutes']
            season_summary['heures_moyennes'] = (season_summary['moyenne_minutes'] / 60).round(2)
            
            return {
                'seasonal_details': seasonal_data.to_dict('records'),
                'season_summary': season_summary.to_dict('records')
            }
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse de la saisonnalité : {e}")
            return {
                'seasonal_details': [],
                'season_summary': []
            }

    def analyser_efficacite_services(self, selected_year=None):
        """Advanced service efficiency analysis"""
        try:
            df_filtered = self.get_filtered_data(selected_year)
            if df_filtered.empty:
                return []

            service_efficiency = df_filtered.groupby('libelle_y').agg({
                'mat_pers': 'nunique',
                'duree_minutes': ['sum', 'mean', 'std'],
                'nbr_pointages': 'sum',
                'date_point': 'count'
            }).reset_index()
            
            service_efficiency.columns = ['service', 'nb_employes', 'total_minutes', 'moyenne_minutes', 
                                        'std_minutes', 'total_pointages_jour', 'total_records']
            
            # Handle NaN values in std_minutes
            service_efficiency['std_minutes'] = service_efficiency['std_minutes'].fillna(0)
            service_efficiency['moyenne_minutes'] = service_efficiency['moyenne_minutes'].fillna(0)
            
            # Calculate efficiency metrics
            service_efficiency['heures_moyennes'] = (service_efficiency['moyenne_minutes'] / 60).round(2)
            
            # Avoid division by zero for coefficient_variation
            service_efficiency['coefficient_variation'] = np.where(
                service_efficiency['moyenne_minutes'] > 0,
                (service_efficiency['std_minutes'] / service_efficiency['moyenne_minutes'] * 100).round(2),
                0
            )
            
            service_efficiency['pointages_par_employe'] = np.where(
                service_efficiency['nb_employes'] > 0,
                (service_efficiency['total_pointages_jour'] / service_efficiency['nb_employes']).round(2),
                0
            )
            
            # Productivity score (normalized)
            max_hours = service_efficiency['heures_moyennes'].max()
            service_efficiency['score_productivite'] = np.where(
                max_hours > 0,
                (service_efficiency['heures_moyennes'] / max_hours * 100).round(2),
                0
            )
            
            return service_efficiency.to_dict('records')
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse de l'efficacité des services : {e}")
            return []

    def analyser_presence_services(self, selected_year=None):
        """Analyse attendance coverage by service, including employees without pointage."""
        try:
            df_filtered = self.get_filtered_data(selected_year)

            if self.df_employe is None or len(self.df_employe) == 0:
                return []

            statut_series = self.df_employe.get('statut', pd.Series('', index=self.df_employe.index)).fillna('').astype(str).str.lower()
            employes_actifs = self.df_employe[~statut_series.isin(['parti/démissionné', 'parti', 'démissionné', 'demissionne', 'depart'])].copy()
            employes_actifs['libelle'] = employes_actifs.get('libelle', pd.Series(dtype=str)).fillna('Service Inconnu')
            service_totaux = employes_actifs.groupby('libelle')['mat_pers'].nunique()

            if df_filtered.empty:
                results = []
                for service, total_employes in service_totaux.items():
                    results.append({
                        'service': service,
                        'employes_total': int(total_employes),
                        'employes_pointage': 0,
                        'taux_couverture': 0.0,
                        'jours_pointage_moyens': 0.0,
                        'jours_pointage_total': 0,
                        'employes_sans_pointage': int(total_employes)
                    })
                return results

            df_filtered['libelle_y'] = df_filtered['libelle_y'].fillna('Service Inconnu')

            df_valid = df_filtered[df_filtered['a_duree_valide']]

            employes_pointage = df_filtered.groupby('libelle_y')['mat_pers'].nunique()
            employes_pointage_valides = df_valid.groupby('libelle_y')['mat_pers'].nunique()

            jours_pointage = df_valid.groupby(['libelle_y', 'mat_pers'])['date_point'].nunique().groupby('libelle_y').mean()
            jours_pointage_total = df_valid.groupby('libelle_y')['date_point'].nunique()

            services = sorted(set(service_totaux.index.tolist()) | set(employes_pointage.index.tolist()))

            resultats = []
            for service in services:
                total_employes = int(service_totaux.get(service, 0))
                pointage_employes = int(employes_pointage.get(service, 0))
                pointage_valides = int(employes_pointage_valides.get(service, 0))
                taux_couverture = (pointage_valides / total_employes) if total_employes > 0 else 0
                jours_pointage_moyens = float(jours_pointage.get(service, 0.0))
                jours_pointage_total_service = int(jours_pointage_total.get(service, 0))
                employes_sans_pointage = max(total_employes - pointage_employes, 0)

                resultats.append({
                    'service': service,
                    'employes_total': total_employes,
                    'employes_pointage': pointage_employes,
                    'employes_pointage_valides': pointage_valides,
                    'taux_couverture': round(taux_couverture, 3),
                    'jours_pointage_moyens': round(jours_pointage_moyens, 1),
                    'jours_pointage_total': jours_pointage_total_service,
                    'employes_sans_pointage': employes_sans_pointage
                })

            resultats.sort(key=lambda x: x['taux_couverture'], reverse=True)
            return resultats
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse de la présence par service : {e}")
            return []

    def calculer_kpi_taux_adoption(self, selected_year=None):
        try:
            df_filtered = self.get_filtered_data(selected_year)
            if df_filtered.empty:
                return []

            pointants_par_service = df_filtered.groupby('libelle_y')['mat_pers'].nunique().reset_index(name='nb_pointants')
            total_employes_par_service = self.df_employe[
                ~self.df_employe['statut'].isin(['Parti/Démissionné'])
            ].groupby('libelle')['mat_pers'].nunique().reset_index(name='nb_total_employes')
            
            kpi_adoption = pointants_par_service.merge(
                total_employes_par_service, left_on='libelle_y', right_on='libelle', how='left'
            )
            
            # Fill missing values and calculate adoption rate
            kpi_adoption['nb_total_employes'] = kpi_adoption['nb_total_employes'].fillna(kpi_adoption['nb_pointants'])
            kpi_adoption['taux_adoption_pointage'] = np.where(
                kpi_adoption['nb_total_employes'] > 0,
                (kpi_adoption['nb_pointants'] / kpi_adoption['nb_total_employes'] * 100).round(2),
                0
            )
            
            return kpi_adoption.to_dict(orient='records')
        except Exception as e:
            logging.error(f"Erreur lors du calcul du KPI d'adoption : {e}")
            return []

    def calculer_kpis_assiduite(self, selected_year=None):
        try:
            df_filtered = self.get_filtered_data(selected_year)
            if df_filtered.empty:
                return {
                    'assiduite_quotidienne_moyenne': [],
                    'heures_travail_moyennes': [],
                    'analyse_heures_supplementaires': [],
                    'heures_de_pointe': [],
                    'modele_hebdomadaire': []
                }

            kpis = {}
            
            # Daily average attendance
            assiduite_moyenne = df_filtered.groupby(['libelle_y', 'date_point']).agg({
                'mat_pers': 'nunique'
            }).reset_index()
            kpis['assiduite_quotidienne_moyenne'] = assiduite_moyenne.groupby('libelle_y')['mat_pers'].mean().reset_index()
            kpis['assiduite_quotidienne_moyenne'].columns = ['service', 'moyenne_employes']
            kpis['assiduite_quotidienne_moyenne'] = kpis['assiduite_quotidienne_moyenne'].to_dict('records')
            
            # Average working hours
            heures_moyennes = df_filtered.groupby(['libelle_y', 'mat_pers'])['duree_minutes'].mean().reset_index()
            heures_moyennes['duree_heures'] = heures_moyennes['duree_minutes'] / 60
            kpis['heures_travail_moyennes'] = heures_moyennes.groupby('libelle_y')['duree_heures'].mean().reset_index()
            kpis['heures_travail_moyennes'].columns = ['service', 'heures_moyennes_par_jour']
            kpis['heures_travail_moyennes']['heures_moyennes_par_jour'] = kpis['heures_travail_moyennes']['heures_moyennes_par_jour'].round(2)
            kpis['heures_travail_moyennes'] = kpis['heures_travail_moyennes'].to_dict('records')
            
            # Overtime analysis
            seuil_heures_sup = 8 * 60
            donnees_heures_sup = df_filtered.copy()
            donnees_heures_sup['minutes_heures_sup'] = np.maximum(0, donnees_heures_sup['duree_minutes'] - seuil_heures_sup)
            donnees_heures_sup['a_heures_sup'] = donnees_heures_sup['minutes_heures_sup'] > 0
            
            overtime_analysis = donnees_heures_sup.groupby('libelle_y').agg({
                'minutes_heures_sup': 'sum',
                'a_heures_sup': 'sum',
                'mat_pers': 'count'
            }).reset_index()
            
            overtime_analysis['taux_heures_sup'] = np.where(
                overtime_analysis['mat_pers'] > 0,
                (overtime_analysis['a_heures_sup'] / overtime_analysis['mat_pers'] * 100).round(2),
                0
            )
            overtime_analysis['heures_sup_moyennes'] = (overtime_analysis['minutes_heures_sup'] / 60).round(2)
            overtime_analysis.columns = ['libelle_y', 'minutes_heures_sup', 'a_heures_sup', 'mat_pers', 'taux_heures_sup', 'heures_sup_moyennes']
            kpis['analyse_heures_supplementaires'] = overtime_analysis.to_dict('records')
            
            # Peak hours
            try:
                heures_df = df_filtered.copy()
                heures_df['heure_entree_parsee'] = pd.to_datetime(
                    heures_df['heure_entree'], format='%H:%M', errors='coerce'
                ).dt.hour
                peak_hours = heures_df.dropna(subset=['heure_entree_parsee']).groupby('heure_entree_parsee')['mat_pers'].count().reset_index()
                peak_hours.columns = ['heure_entree_parsee', 'mat_pers']
                kpis['heures_de_pointe'] = peak_hours.to_dict('records')
            except:
                kpis['heures_de_pointe'] = []
            
            # Weekly patterns
            weekly_pattern = df_filtered.groupby('jour_semaine')['mat_pers'].count().reset_index()
            weekly_pattern.columns = ['jour_semaine', 'mat_pers']
            kpis['modele_hebdomadaire'] = weekly_pattern.to_dict('records')
            
            return kpis
        except Exception as e:
            logging.error(f"Erreur lors du calcul des KPIs d'assiduité : {e}")
            return {
                'assiduite_quotidienne_moyenne': [],
                'heures_travail_moyennes': [],
                'analyse_heures_supplementaires': [],
                'heures_de_pointe': [],
                'modele_hebdomadaire': []
            }

    def generer_stats_resume(self, selected_year=None, force_all_years=False):
        """Generate summary statistics"""
        try:
            if self.df_fusionne is None or self.df_fusionne.empty:
                return {
                    'total_pointages': 0,
                    'total_pointages_annee_courante': 0,
                    'employes_uniques': 0,
                    'employes_uniques_total': 0,
                    'employes_actifs_annee_courante': 0,
                    'services_actifs': 0,
                    'avg_daily_hours': 0,
                    'avg_daily_hours_current_year': 0,
                    'total_work_hours': 0,
                    'total_work_hours_selected': 0,
                    'annees_couvertes': [],
                    'selected_year': None if force_all_years else selected_year
                }

            df_all = self.df_fusionne.copy()
            df_all = df_all.dropna(subset=['annee_pointage'])
            annees_couvertes = sorted(df_all['annee_pointage'].unique().tolist())

            if force_all_years:
                year_for_focus = None
            elif selected_year and selected_year in annees_couvertes:
                year_for_focus = selected_year
            else:
                year_for_focus = annees_couvertes[-1] if annees_couvertes else None

            if force_all_years or year_for_focus is None:
                current_year_data = df_all
            else:
                current_year_data = df_all[df_all['annee_pointage'] == year_for_focus]
            
            stats = {
                'total_pointages': int(self.df_fusionne['nbr_pointages'].sum()),
                'total_pointages_annee_courante': int(current_year_data['nbr_pointages'].sum()),
                'employes_uniques': int(current_year_data['mat_pers'].nunique()) if len(current_year_data) > 0 else 0,
                'employes_uniques_total': int(self.df_fusionne['mat_pers'].nunique()),
                'employes_actifs_annee_courante': int(current_year_data['mat_pers'].nunique()) if len(current_year_data) > 0 else 0,
                'services_actifs': int(current_year_data['libelle_y'].nunique()) if len(current_year_data) > 0 else 0,
                'avg_daily_hours': round(self.df_fusionne['duree_minutes'].mean() / 60, 2) if len(self.df_fusionne) > 0 else 0,
                'avg_daily_hours_current_year': round(current_year_data['duree_minutes'].mean() / 60, 2) if len(current_year_data) > 0 else 0,
                'total_work_hours': int(self.df_fusionne['duree_minutes'].sum() / 60),
                'total_work_hours_selected': int(current_year_data['duree_minutes'].sum() / 60) if len(current_year_data) > 0 else 0,
                'annees_couvertes': annees_couvertes,
                'selected_year': None if force_all_years else year_for_focus
            }
            return stats
        except Exception as e:
            logging.error(f"Erreur lors de la génération des statistiques résumées : {e}")
            return {
                'total_pointages': 0,
                'total_pointages_annee_courante': 0,
                'employes_uniques': 0,
                'employes_uniques_total': 0,
                'employes_actifs_annee_courante': 0,
                'services_actifs': 0,
                'avg_daily_hours': 0,
                'avg_daily_hours_current_year': 0,
                'total_work_hours': 0,
                'total_work_hours_selected': 0,
                'annees_couvertes': [],
                'selected_year': None if force_all_years else selected_year
            }

# Instance globale
analyseur = AnalyseurPointageRH()

@hr_analytics_bp.route('/hr_analytics')
def hr_analytics_dashboard():
    """Render the HR analytics dashboard"""
    return render_template('hr_analytics.html')

@hr_analytics_bp.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    try:
        selected_year = request.args.get('year', type=int)
        mode = request.args.get('mode')
        force_all_years = mode == 'all'

        if force_all_years:
            selected_year = None

        if not analyseur.charger_donnees():
            return jsonify({'error': 'Échec du chargement des données depuis la base de données'}), 500
            
        if not analyseur.nettoyer_et_preparer_donnees():
            return jsonify({'error': 'Échec du traitement des données'}), 500
        
        data = {
            'assiduite': analyseur.calculer_kpis_assiduite(selected_year),
            'summary': analyseur.generer_stats_resume(selected_year, force_all_years=force_all_years),
            'yearly_trends': analyseur.calculer_tendances_annuelles(selected_year),
            'employee_categories': analyseur.analyser_categories_employes(selected_year),
            'seasonality': analyseur.analyser_saisonnalite(selected_year),
            'service_efficiency': analyseur.analyser_efficacite_services(selected_year)
        }
        
        return jsonify(data)
        
    except Exception as e:
        logging.error(f"Erreur API : {e}")
        return jsonify({'error': str(e)}), 500
