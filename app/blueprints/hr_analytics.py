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

from app.config.database import get_oracle_credentials, build_sqlalchemy_url

warnings.filterwarnings('ignore')

# Create HR analytics blueprint
hr_analytics_bp = Blueprint('hr_analytics', __name__, template_folder='templates')
CORS(hr_analytics_bp)

class AnalyseurPointageRH:
    def __init__(self, credentials=None):
        credentials = credentials or get_oracle_credentials()
        self.chaine_connexion_oracle = build_sqlalchemy_url(credentials)
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

            self.df_temps = pd.read_sql("SELECT * FROM DIM_DATE", self.moteur)
            self.df_employe = pd.read_sql("SELECT * FROM DIM_PERSONNEL", self.moteur)
            self.df_pointage = pd.read_sql("SELECT * FROM FAIT_TEMPS_PRESENCE", self.moteur)
            self.df_service = pd.read_sql("SELECT * FROM DIM_SERVICE", self.moteur)
            self.df_categories = pd.read_sql("SELECT * FROM DIM_CATEGORIE", self.moteur)

            for df in [self.df_temps, self.df_employe, self.df_pointage, self.df_service, self.df_categories]:
                df.columns = df.columns.str.lower().str.strip()

            if 'cod_c' in self.df_categories.columns and 'cod_cat' not in self.df_categories.columns:
                self.df_categories.rename(columns={'cod_c': 'cod_cat'}, inplace=True)

            if 'libelle' in self.df_service.columns and 'libelle_service' not in self.df_service.columns:
                self.df_service.rename(columns={'libelle': 'libelle_service'}, inplace=True)

            return True
        except Exception as e:
            logging.error(f"Erreur lors du chargement des données : {e}")
            return False

    def nettoyer_et_preparer_donnees(self):
        try:
            if self.df_pointage is None or self.df_pointage.empty:
                logging.error("Aucune donnée de présence à préparer")
                self.df_fusionne = pd.DataFrame()
                return False

            df_presence = self.df_pointage.copy()

            if 'heures_travaillees' not in df_presence.columns:
                df_presence['heures_travaillees'] = 0
            df_presence['heures_travaillees'] = pd.to_numeric(
                df_presence['heures_travaillees'], errors='coerce'
            ).fillna(0)

            if 'presence_flag' not in df_presence.columns:
                df_presence['presence_flag'] = 0
            presence_flag = pd.to_numeric(
                df_presence['presence_flag'], errors='coerce'
            ).fillna(0)
            df_presence['presence_flag'] = presence_flag.astype(int)
            df_presence['nbr_pointages'] = np.where(df_presence['presence_flag'] > 0, 1, 0)
            df_presence['duree_minutes'] = (df_presence['heures_travaillees'] * 60).round(2)
            df_presence['duree_minutes'] = df_presence['duree_minutes'].clip(lower=0)
            df_presence['a_duree_valide'] = df_presence['duree_minutes'] > 0
            df_presence['est_present'] = df_presence['presence_flag'] > 0

            for col in ['heure_entree', 'heure_sortie']:
                if col not in df_presence.columns:
                    df_presence[col] = pd.NA

            df_merged = df_presence.merge(
                self.df_temps, on='id_temps', how='left', suffixes=('', '_date')
            ).merge(
                self.df_employe, on='mat_pers', how='left', suffixes=('', '_emp')
            ).merge(
                self.df_service, left_on='cod_serv', right_on='code_serv', how='left'
            )

            if self.df_categories is not None and not self.df_categories.empty:
                df_merged = df_merged.merge(self.df_categories, on='cod_cat', how='left', suffixes=('', '_cat'))

            date_column = None
            for candidate in ['date_jour', 'date_comp', 'date_point']:
                if candidate in df_merged.columns:
                    date_column = candidate
                    break

            if date_column:
                df_merged['date_point'] = pd.to_datetime(df_merged[date_column], errors='coerce')
            else:
                df_merged['date_point'] = pd.NaT

            if 'annee' in df_merged.columns:
                df_merged['annee_pointage'] = pd.to_numeric(df_merged['annee'], errors='coerce')
            else:
                df_merged['annee_pointage'] = df_merged['date_point'].dt.year

            if 'mois' in df_merged.columns:
                df_merged['mois'] = pd.to_numeric(df_merged['mois'], errors='coerce')
            elif 'mois_numero_global' in df_merged.columns:
                df_merged['mois'] = pd.to_numeric(df_merged['mois_numero_global'], errors='coerce')
            elif 'mois_rang' in df_merged.columns:
                df_merged['mois'] = pd.to_numeric(df_merged['mois_rang'], errors='coerce')

            if 'jour_semaine' not in df_merged.columns:
                if 'jour_semaine_clean' in df_merged.columns:
                    df_merged['jour_semaine'] = df_merged['jour_semaine_clean']
                elif 'nom_jour' in df_merged.columns:
                    df_merged['jour_semaine'] = df_merged['nom_jour']

            if 'libelle_service' in df_merged.columns:
                df_merged['libelle_y'] = df_merged['libelle_service']
            elif 'libelle' in df_merged.columns:
                df_merged['libelle_y'] = df_merged['libelle']
            else:
                df_merged['libelle_y'] = 'Service Inconnu'
            df_merged['libelle_y'] = df_merged['libelle_y'].fillna('Service Inconnu')

            if 'statut' in df_merged.columns:
                statut_series = df_merged['statut'].fillna('').astype(str).str.lower()
                df_merged = df_merged[~statut_series.isin(['parti/démissionné', 'parti', 'démissionné', 'demissionne', 'depart'])]

            df_merged['presence_flag'] = pd.to_numeric(
                df_merged.get('presence_flag', 0), errors='coerce'
            ).fillna(0).astype(int)
            df_merged['nbr_pointages'] = np.where(df_merged['presence_flag'] > 0, 1, 0)
            df_merged['a_duree_valide'] = df_merged['a_duree_valide'].fillna(False)

            if 'jour_semaine' in df_merged.columns:
                df_merged['jour_semaine'] = df_merged['jour_semaine'].fillna('Inconnu')
            else:
                df_merged['jour_semaine'] = 'Inconnu'

            self.df_fusionne = df_merged
            return True
        except Exception as e:
            logging.error(f"Erreur lors du nettoyage des données : {e}")
            return False

    def get_filtered_data(self, selected_year=None, filters=None):
        """Return a filtered copy of the fused dataframe for the requested year."""
        if self.df_fusionne is None:
            return pd.DataFrame()
        df = self.df_fusionne.copy()
        filters = filters or {}

        service_filter = (filters.get('service') or '').strip()
        category_filter = (filters.get('category') or '').strip()
        month_range = filters.get('month_range')

        if service_filter:
            df['libelle_y'] = df.get('libelle_y', pd.Series(dtype=str)).fillna('Service Inconnu')
            df = df[df['libelle_y'].str.lower() == service_filter.lower()]

        if category_filter:
            category_cols = ['categorie', 'lib_cat', 'libelle_cat', 'libelle_categorie']
            cat_series = None
            for col in category_cols:
                if col in df.columns:
                    cat_series = df[col].astype(str)
                    break
            if cat_series is None and 'cod_cat' in df.columns and self.df_categories is not None:
                df = df.merge(self.df_categories[['cod_cat', 'lib_cat']].rename(columns={'lib_cat': 'categorie_temp'}), on='cod_cat', how='left')
                cat_series = df['categorie_temp']
            if cat_series is not None:
                df = df[cat_series.fillna('').str.lower() == category_filter.lower()]

        if month_range and df.shape[0] > 0:
            month_start, month_end = month_range
            if month_start or month_end:
                month_start = int(month_start or 1)
                month_end = int(month_end or 12)
                if month_start < 1: month_start = 1
                if month_end > 12: month_end = 12
                if month_end < month_start:
                    month_start, month_end = month_end, month_start

                if 'mois' not in df.columns:
                    if 'date_point' in df.columns:
                        df['mois'] = df['date_point'].dt.month
                    else:
                        df['mois'] = pd.to_numeric(df.get('mois'), errors='coerce')

                df['mois'] = pd.to_numeric(df.get('mois'), errors='coerce')
                df = df[(df['mois'] >= month_start) & (df['mois'] <= month_end)]

        if selected_year:
            return df[df['annee_pointage'] == selected_year].copy()
        return df

    def calculer_tendances_annuelles(self, selected_year=None, filters=None):
        """Calculate yearly trends and drilldowns based on the selected year."""
        try:
            df_all = self.get_filtered_data(None, filters)
            if df_all is None or df_all.empty:
                return {
                    'yearly_overview': [],
                    'monthly_current_year': [],
                    'service_yearly_performance': [],
                    'selected_year': selected_year
                }

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

    def analyser_categories_employes(self, selected_year=None, filters=None):
        """Analyze employee categories performance"""
        try:
            if self.df_categories is not None and len(self.df_categories) > 0:
                df_filtered = self.get_filtered_data(selected_year, filters)
                if df_filtered.empty:
                    return []

                # Merge with categories
                df_with_cat = df_filtered.copy()
                if 'cod_cat' in df_with_cat.columns and 'cod_cat' in self.df_categories.columns:
                    df_with_cat = df_with_cat.merge(
                        self.df_categories, on='cod_cat', how='left', suffixes=('', '_dimcat')
                    )

                categorie_col = None
                for candidate in ['lib_cat', 'libelle_cat', 'categorie', 'libelle_categorie', 'libelle']:
                    if candidate in df_with_cat.columns:
                        categorie_col = candidate
                        break

                if categorie_col is None:
                    logging.warning("Colonne de catégorie introuvable pour l'analyse des catégories")
                    return []

                cat_analysis = df_with_cat.groupby(categorie_col).agg({
                    'mat_pers': 'nunique',
                    'duree_minutes': ['sum', 'mean'],
                    'nbr_pointages': 'sum'
                }).reset_index()

                cat_analysis.columns = [
                    'categorie', 'nb_employes', 'total_minutes',
                    'moyenne_minutes', 'total_pointages'
                ]
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

    def analyser_saisonnalite(self, selected_year=None, filters=None):
        """Analyze seasonal patterns"""
        try:
            df_filtered = self.get_filtered_data(selected_year, filters).copy()
            if df_filtered.empty:
                return {
                    'seasonal_details': [],
                    'season_summary': []
                }

            mois_col = None
            for candidate in ['mois', 'mois_numero_global', 'mois_rang', 'month']:
                if candidate in df_filtered.columns:
                    mois_col = candidate
                    break

            if mois_col is None and 'date_point' in df_filtered.columns:
                df_filtered['mois_calcule'] = df_filtered['date_point'].dt.month
                if df_filtered['mois_calcule'].notna().any():
                    mois_col = 'mois_calcule'

            if mois_col is None:
                logging.warning("Colonne mois introuvable pour l'analyse de saisonnalité")
                return {
                    'seasonal_details': [],
                    'season_summary': []
                }

            df_filtered[mois_col] = pd.to_numeric(df_filtered[mois_col], errors='coerce')

            saison_col = None
            for candidate in ['saison', 'saison_arabe', 'season', 'saison_clean']:
                if candidate in df_filtered.columns:
                    saison_col = candidate
                    break

            if saison_col is None:
                # Derive a season grouping from the month number
                def mois_vers_saison(mois):
                    if pd.isna(mois):
                        return 'Inconnu'
                    mois_int = int(mois)
                    if mois_int in (12, 1, 2):
                        return 'Hiver'
                    if mois_int in (3, 4, 5):
                        return 'Printemps'
                    if mois_int in (6, 7, 8):
                        return 'Été'
                    if mois_int in (9, 10, 11):
                        return 'Automne'
                    return 'Inconnu'

                df_filtered['saison_calculee'] = df_filtered[mois_col].apply(mois_vers_saison)
                saison_col = 'saison_calculee'

            seasonal_data = df_filtered.groupby([saison_col, mois_col]).agg({
                'mat_pers': 'nunique',
                'duree_minutes': ['sum', 'mean']
            }).reset_index()
            
            seasonal_data.columns = [
                'saison', 'mois', 'employes_actifs',
                'total_minutes', 'moyenne_minutes'
            ]
            seasonal_data['total_heures'] = (seasonal_data['total_minutes'] / 60).round(0).astype(int)
            seasonal_data['heures_moyennes'] = (seasonal_data['moyenne_minutes'] / 60).round(2)
            
            # Season summary
            season_summary = df_filtered.groupby(saison_col).agg({
                'mat_pers': 'nunique',
                'duree_minutes': ['sum', 'mean']
            }).reset_index()
            season_summary.columns = [
                'saison', 'employes_moyens',
                'total_minutes', 'moyenne_minutes'
            ]
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

    def analyser_efficacite_services(self, selected_year=None, filters=None):
        """Advanced service efficiency analysis"""
        try:
            df_filtered = self.get_filtered_data(selected_year, filters)
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

    def analyser_presence_services(self, selected_year=None, filters=None):
        """Analyse attendance coverage by service, including employees without pointage."""
        try:
            df_filtered = self.get_filtered_data(selected_year, filters)

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

    def calculer_kpi_taux_adoption(self, selected_year=None, filters=None):
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

    def calculer_kpis_assiduite(self, selected_year=None, filters=None):
        try:
            df_filtered = self.get_filtered_data(selected_year, filters)
            if df_filtered.empty:
                return {
                    'assiduite_quotidienne_moyenne': [],
                    'heures_travail_moyennes': [],
                    'analyse_heures_supplementaires': [],
                    'heures_de_pointe': [],
                    'heures_de_pointe_sortie': [],
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
                heure_source = None
                for candidate in ['heure_entree', 'heure', 'heure_debut', 'heure_arrivee']:
                    if candidate in heures_df.columns:
                        heure_source = candidate
                        break

                if heure_source is not None:
                    heures_df['heure_entree_decimal'] = pd.to_numeric(
                        heures_df[heure_source], errors='coerce'
                    )
                    heures_df['heure_entree_parsee'] = np.floor(
                        heures_df['heure_entree_decimal']
                    ).astype('Int64')
                    peak_hours = (
                        heures_df.dropna(subset=['heure_entree_parsee'])
                        .groupby('heure_entree_parsee')['mat_pers']
                        .nunique()
                        .reset_index()
                    )
                    peak_hours.columns = ['heure', 'nombre_entrees']
                    peak_hours['heure'] = peak_hours['heure'].astype(int)
                    peak_hours['nombre_entrees'] = peak_hours['nombre_entrees'].astype(int)
                    kpis['heures_de_pointe'] = peak_hours.to_dict('records')
                else:
                    kpis['heures_de_pointe'] = []
            except Exception as error:
                logging.warning(f"Impossible de calculer les heures de pointe : {error}")
                kpis['heures_de_pointe'] = []

            try:
                heures_sortie_df = df_filtered.copy()
                sortie_source = None
                for candidate in ['heure_sortie', 'heure_fin', 'heure_depart']:
                    if candidate in heures_sortie_df.columns:
                        sortie_source = candidate
                        break

                if sortie_source is not None:
                    heures_sortie_df['heure_sortie_decimal'] = pd.to_numeric(
                        heures_sortie_df[sortie_source], errors='coerce'
                    )
                    heures_sortie_df['heure_sortie_parsee'] = np.floor(
                        heures_sortie_df['heure_sortie_decimal']
                    ).astype('Int64')
                    peak_hours_sortie = (
                        heures_sortie_df.dropna(subset=['heure_sortie_parsee'])
                        .groupby('heure_sortie_parsee')['mat_pers']
                        .nunique()
                        .reset_index()
                    )
                    peak_hours_sortie.columns = ['heure', 'nombre_sorties']
                    peak_hours_sortie['heure'] = peak_hours_sortie['heure'].astype(int)
                    peak_hours_sortie['nombre_sorties'] = peak_hours_sortie['nombre_sorties'].astype(int)
                    kpis['heures_de_pointe_sortie'] = peak_hours_sortie.to_dict('records')
                else:
                    kpis['heures_de_pointe_sortie'] = []
            except Exception as error:
                logging.warning(f"Impossible de calculer les heures de sortie : {error}")
                kpis['heures_de_pointe_sortie'] = []
            
            # Weekly patterns
            weekly_pattern = (
                df_filtered.groupby('jour_semaine')['mat_pers']
                .nunique()
                .reset_index()
            )
            weekly_pattern.columns = ['jour_semaine', 'nombre_presences']
            weekly_pattern['jour_semaine'] = weekly_pattern['jour_semaine'].fillna('Inconnu')
            weekly_pattern['nombre_presences'] = weekly_pattern['nombre_presences'].astype(int)
            kpis['modele_hebdomadaire'] = weekly_pattern.to_dict('records')
            
            return kpis
        except Exception as e:
            logging.error(f"Erreur lors du calcul des KPIs d'assiduité : {e}")
            return {
                'assiduite_quotidienne_moyenne': [],
                'heures_travail_moyennes': [],
                'analyse_heures_supplementaires': [],
                'heures_de_pointe': [],
                'heures_de_pointe_sortie': [],
                'modele_hebdomadaire': []
            }

    def analyser_presence_individuelle(self, selected_year=None, months=3, top_n=50):
        """Retourne les employés les plus réguliers/irréguliers sur la période analysée."""
        try:
            df_filtered = self.get_filtered_data(selected_year).copy()
            if df_filtered.empty:
                return {
                    'top_regular': [],
                    'top_irregular': [],
                    'service_breakdown': [],
                    'available_services': [],
                    'context': {}
                }

            # Assurer une colonne date exploitable
            date_candidates = ['date_point', 'date_jour', 'date_comp', 'date', 'date_pointage']
            date_series = None
            for candidate in date_candidates:
                if candidate in df_filtered.columns:
                    tentative = pd.to_datetime(df_filtered[candidate], errors='coerce')
                    if tentative.notna().any():
                        date_series = tentative
                        break

            if date_series is None:
                # Essayer de reconstruire une date à partir des composantes année/mois/jour
                annee_raw = df_filtered.get('annee_pointage')
                annee = pd.to_numeric(annee_raw, errors='coerce') if annee_raw is not None else pd.Series(np.nan, index=df_filtered.index)

                mois_source = None
                for candidate in ['mois', 'mois_numero_global', 'mois_rang']:
                    if candidate in df_filtered.columns:
                        mois_source = df_filtered[candidate]
                        break
                mois = pd.to_numeric(mois_source, errors='coerce') if mois_source is not None else pd.Series(1, index=df_filtered.index)

                jour_source = None
                for candidate in ['jour_mois', 'jour_numero', 'jour']:
                    if candidate in df_filtered.columns:
                        jour_source = df_filtered[candidate]
                        break
                jour = pd.to_numeric(jour_source, errors='coerce') if jour_source is not None else pd.Series(np.nan, index=df_filtered.index)

                clean_years = annee.dropna()
                base_year = int(clean_years.iloc[0]) if not clean_years.empty else pd.Timestamp.today().year
                year_vals = annee.fillna(base_year).round().astype(int)

                month_vals = mois.fillna(1).round().astype(int)
                day_vals = jour.fillna(1).round().astype(int)

                composed = pd.to_datetime(
                    dict(year=year_vals, month=month_vals, day=day_vals),
                    errors='coerce'
                )
                if composed.notna().any():
                    date_series = composed

            if date_series is None:
                logging.warning("Aucune colonne de date exploitable pour l'analyse individuelle.")
                return {
                    'top_regular': [],
                    'top_irregular': [],
                    'service_breakdown': [],
                    'available_services': [],
                    'context': {}
                }

            df_filtered['date_point_calculee'] = date_series
            df_filtered = df_filtered.dropna(subset=['date_point_calculee'])
            df_filtered['date_point'] = df_filtered['date_point_calculee']
            df_filtered = df_filtered.drop(columns=['date_point_calculee'])
            if df_filtered.empty:
                return {
                    'top_regular': [],
                    'top_irregular': [],
                    'service_breakdown': [],
                    'available_services': [],
                    'context': {}
                }

            df_filtered = df_filtered.dropna(subset=['mat_pers'])

            df_filtered['libelle_y'] = df_filtered.get('libelle_y', pd.Series(dtype=str)).fillna('Service Inconnu')

            # Déterminer la fenêtre temporelle (dernier X mois)
            last_date = df_filtered['date_point'].max()
            if pd.notna(last_date) and months:
                start_threshold = last_date - pd.DateOffset(months=months)
                recent_df = df_filtered[df_filtered['date_point'] >= start_threshold]
                if not recent_df.empty:
                    df_filtered = recent_df

            period_start = df_filtered['date_point'].min()
            period_end = df_filtered['date_point'].max()

            if pd.isna(period_start) or pd.isna(period_end):
                return {
                    'top_regular': [],
                    'top_irregular': [],
                    'service_breakdown': [],
                    'available_services': [],
                    'context': {}
                }

            df_filtered['jour_pointage'] = df_filtered['date_point'].dt.normalize()
            business_days = pd.bdate_range(start=period_start, end=period_end)
            total_business_days = len(business_days)
            if total_business_days == 0:
                total_business_days = max(int(df_filtered['jour_pointage'].nunique()), 1)

            # Construire un identifiant lisible
            nom_candidates = [
                'nom_prenom', 'nom_complet', 'nom', 'nom_personnel', 'nom_emp', 'nom_pers'
            ]
            prenom_candidates = ['prenom', 'prenoms', 'prenom_emp']

            nom_col = next((col for col in nom_candidates if col in df_filtered.columns), None)
            prenom_col = next((col for col in prenom_candidates if col in df_filtered.columns), None)

            if nom_col and prenom_col and nom_col != prenom_col:
                df_filtered['nom_affiche'] = (
                    df_filtered[prenom_col].fillna('') + ' ' + df_filtered[nom_col].fillna('')
                ).str.strip()
            elif nom_col:
                df_filtered['nom_affiche'] = df_filtered[nom_col].fillna('')
            elif prenom_col:
                df_filtered['nom_affiche'] = df_filtered[prenom_col].fillna('')
            else:
                df_filtered['nom_affiche'] = ''

            df_filtered['nom_affiche'] = df_filtered['nom_affiche'].astype(str)
            mask_nom_vide = df_filtered['nom_affiche'].str.strip() == ''
            if mask_nom_vide.any():
                df_filtered.loc[mask_nom_vide, 'nom_affiche'] = (
                    'Employé ' + df_filtered.loc[mask_nom_vide, 'mat_pers'].astype(str)
                )

            target_daily_hours = 8
            df_filtered['duree_minutes'] = pd.to_numeric(df_filtered.get('duree_minutes', 0), errors='coerce').fillna(0)
            df_filtered['nbr_pointages'] = pd.to_numeric(df_filtered.get('nbr_pointages', 0), errors='coerce').fillna(0)

            group_cols = ['mat_pers', 'libelle_y', 'nom_affiche']
            employee_stats = (
                df_filtered.groupby(group_cols).agg(
                    total_minutes=('duree_minutes', 'sum'),
                    avg_minutes=('duree_minutes', 'mean'),
                    jours_pointes=('jour_pointage', 'nunique'),
                    total_pointages=('nbr_pointages', 'sum')
                )
                .reset_index()
            )

            if employee_stats.empty:
                return {
                    'top_regular': [],
                    'top_irregular': [],
                    'service_breakdown': [],
                    'available_services': [],
                    'context': {}
                }

            employee_stats.rename(columns={'libelle_y': 'service', 'nom_affiche': 'employe'}, inplace=True)

            employee_stats['avg_daily_hours'] = (employee_stats['avg_minutes'] / 60).round(2)
            employee_stats['heures_totales'] = (employee_stats['total_minutes'] / 60).round(1)
            employee_stats['jours_sans_pointage'] = np.maximum(total_business_days - employee_stats['jours_pointes'], 0)
            employee_stats['taux_assiduite'] = np.where(
                total_business_days > 0,
                (employee_stats['jours_pointes'] / total_business_days * 100).round(1),
                0
            )
            employee_stats['retard_vs_objectif'] = (target_daily_hours - employee_stats['avg_daily_hours']).round(2)
            employee_stats['ecart_normalise'] = np.clip(
                1 - (employee_stats['retard_vs_objectif'].abs() / max(target_daily_hours, 1)),
                0,
                1
            )
            employee_stats['score_regularite'] = (
                (employee_stats['taux_assiduite'] / 100 * 0.6) + (employee_stats['ecart_normalise'] * 0.4)
            ) * 100
            employee_stats['score_regularite'] = employee_stats['score_regularite'].round(1)

            sort_regular = employee_stats.sort_values(
                ['score_regularite', 'jours_sans_pointage', 'avg_daily_hours'],
                ascending=[False, True, False]
            )
            sort_irregular = employee_stats.sort_values(
                ['score_regularite', 'jours_sans_pointage', 'avg_daily_hours'],
                ascending=[True, False, True]
            )

            top_regular = sort_regular.head(top_n).to_dict('records')
            top_irregular = sort_irregular.head(top_n).to_dict('records')

            service_breakdown = (
                employee_stats.groupby('service').agg({
                    'mat_pers': 'count',
                    'taux_assiduite': 'mean',
                    'jours_sans_pointage': 'sum',
                    'avg_daily_hours': 'mean',
                    'score_regularite': 'mean'
                }).reset_index()
            )
            service_breakdown.rename(columns={'mat_pers': 'nb_employes'}, inplace=True)
            service_breakdown['taux_assiduite'] = service_breakdown['taux_assiduite'].round(1)
            service_breakdown['avg_daily_hours'] = service_breakdown['avg_daily_hours'].round(2)
            service_breakdown['score_regularite'] = service_breakdown['score_regularite'].round(1)

            available_services = sorted(employee_stats['service'].dropna().unique().tolist())
            employees_directory = (
                employee_stats[['mat_pers', 'employe', 'service']]
                .drop_duplicates()
                .sort_values('employe')
                .to_dict('records')
            )

            return {
                'top_regular': top_regular,
                'top_irregular': top_irregular,
                'service_breakdown': service_breakdown.to_dict('records'),
                'available_services': available_services,
                'employees_directory': employees_directory,
                'context': {
                    'period_start': period_start.isoformat(),
                    'period_end': period_end.isoformat(),
                    'months_window': months,
                    'business_days': total_business_days,
                    'target_daily_hours': target_daily_hours
                }
            }
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse de présence individuelle : {e}")
            return {
                'top_regular': [],
                'top_irregular': [],
                'service_breakdown': [],
                'available_services': [],
                'employees_directory': [],
                'context': {}
            }

    def analyser_presence_employe(self, mat_pers, start_date=None, end_date=None):
        """Provide detailed presence timeline for a single employee."""
        try:
            if not mat_pers:
                return {'error': "Le paramètre 'mat_pers' est requis."}

            if self.df_fusionne is None or self.df_fusionne.empty:
                return {'error': "Aucune donnée disponible."}

            df = self.df_fusionne.copy()
            df['mat_pers_str'] = df['mat_pers'].astype(str)
            mat_pers_str = str(mat_pers)
            df_emp = df[df['mat_pers_str'] == mat_pers_str].copy()

            if df_emp.empty:
                return {
                    'mat_pers': mat_pers,
                    'daily': [],
                    'summary': {},
                    'range': {},
                    'status': 'not_found'
                }

            date_candidates = ['date_point', 'date_jour', 'date_comp', 'date', 'date_pointage']
            date_series = None
            for candidate in date_candidates:
                if candidate in df_emp.columns:
                    tentative = pd.to_datetime(df_emp[candidate], errors='coerce')
                    if tentative.notna().any():
                        date_series = tentative
                        break

            if date_series is None or date_series.notna().sum() == 0:
                annee_raw = df_emp.get('annee_pointage')
                annee = pd.to_numeric(annee_raw, errors='coerce') if annee_raw is not None else pd.Series(np.nan, index=df_emp.index)

                mois_source = None
                for candidate in ['mois', 'mois_numero_global', 'mois_rang']:
                    if candidate in df_emp.columns:
                        mois_source = df_emp[candidate]
                        break
                mois = pd.to_numeric(mois_source, errors='coerce') if mois_source is not None else pd.Series(np.nan, index=df_emp.index)

                jour_source = None
                for candidate in ['jour_mois', 'jour_numero', 'jour']:
                    if candidate in df_emp.columns:
                        jour_source = df_emp[candidate]
                        break
                jour = pd.to_numeric(jour_source, errors='coerce') if jour_source is not None else pd.Series(np.nan, index=df_emp.index)

                clean_years = annee.dropna()
                base_year = int(clean_years.iloc[0]) if not clean_years.empty else pd.Timestamp.today().year

                year_vals = annee.fillna(base_year).round().astype(int, errors='ignore')
                month_vals = mois.fillna(1).round().astype(int, errors='ignore')
                day_vals = jour.fillna(1).round().astype(int, errors='ignore')

                composed = pd.to_datetime(
                    dict(year=year_vals, month=month_vals, day=day_vals),
                    errors='coerce'
                )

                if composed.notna().any():
                    date_series = composed

            if date_series is None or date_series.notna().sum() == 0:
                return {
                    'mat_pers': mat_pers,
                    'daily': [],
                    'summary': {},
                    'range': {},
                    'status': 'no_dates'
                }

            df_emp['date_point'] = date_series
            df_emp = df_emp.dropna(subset=['date_point'])

            if df_emp.empty:
                return {
                    'mat_pers': mat_pers,
                    'daily': [],
                    'summary': {},
                    'range': {},
                    'status': 'no_dates'
                }

            available_start = df_emp['date_point'].min()
            available_end = df_emp['date_point'].max()

            start_dt = pd.to_datetime(start_date, errors='coerce') if start_date else available_start
            end_dt = pd.to_datetime(end_date, errors='coerce') if end_date else available_end

            if pd.isna(start_dt) or start_dt < available_start:
                start_dt = available_start
            if pd.isna(end_dt) or end_dt > available_end:
                end_dt = available_end

            if start_dt > end_dt:
                start_dt, end_dt = end_dt, start_dt

            df_emp = df_emp[(df_emp['date_point'] >= start_dt) & (df_emp['date_point'] <= end_dt)].copy()

            if df_emp.empty:
                return {
                    'mat_pers': mat_pers,
                    'daily': [],
                    'summary': {},
                    'range': {
                        'selected_start': start_dt.isoformat(),
                        'selected_end': end_dt.isoformat(),
                        'available_start': available_start.isoformat(),
                        'available_end': available_end.isoformat()
                    },
                    'status': 'no_records_in_range'
                }

            df_emp['duree_minutes'] = pd.to_numeric(df_emp.get('duree_minutes', 0), errors='coerce').fillna(0)
            df_emp['heures'] = (df_emp['duree_minutes'] / 60).round(2)
            overtime_threshold = 8 * 60
            df_emp['heures_sup'] = np.maximum(0, df_emp['duree_minutes'] - overtime_threshold) / 60
            df_emp['jour_pointage'] = df_emp['date_point'].dt.normalize()

            def _get_display_name(frame):
                name_candidates = ['employe', 'nom_affiche', 'nom_prenom', 'nom_complet', 'nom', 'prenom']
                for candidate in name_candidates:
                    if candidate in frame.columns and frame[candidate].notna().any():
                        val = frame[candidate].dropna().astype(str).iloc[0]
                        if val.strip():
                            return val.strip()
                return f"Employé {mat_pers}"

            display_name = _get_display_name(df_emp)
            service = df_emp.get('libelle_y', pd.Series(['Service Inconnu'])).dropna().iloc[-1]

            daily_summary = (
                df_emp.groupby('jour_pointage')
                .agg({
                    'heures': 'sum',
                    'heures_sup': 'sum',
                    'presence_flag': lambda x: int(np.nanmax(x)),
                    'libelle_y': lambda x: x.dropna().iloc[-1] if not x.dropna().empty else service
                })
                .reset_index()
            )

            daily_summary['date'] = daily_summary['jour_pointage'].dt.strftime('%Y-%m-%d')
            daily_summary['heures'] = daily_summary['heures'].round(2)
            daily_summary['heures_sup'] = daily_summary['heures_sup'].round(2)
            daily_records = daily_summary[['date', 'heures', 'heures_sup', 'presence_flag', 'libelle_y']].to_dict('records')

            total_hours = round(df_emp['heures'].sum(), 2)
            avg_hours = round(df_emp['heures'].mean(), 2)
            presence_days = int(daily_summary['jour_pointage'].nunique())
            overtime_hours = round(df_emp['heures_sup'].sum(), 2)
            total_pointages = int(len(df_emp))

            summary = {
                'total_hours': total_hours,
                'average_hours': avg_hours,
                'presence_days': presence_days,
                'overtime_hours': overtime_hours,
                'total_pointages': total_pointages,
                'service': service,
                'employe': display_name
            }

            return {
                'mat_pers': mat_pers,
                'employe': display_name,
                'service': service,
                'summary': summary,
                'daily': daily_records,
                'range': {
                    'selected_start': start_dt.isoformat(),
                    'selected_end': end_dt.isoformat(),
                    'available_start': available_start.isoformat(),
                    'available_end': available_end.isoformat()
                },
                'status': 'ok'
            }
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse de présence d'un employé : {e}")
            return {'error': "Impossible de récupérer les données de présence."}

    def generer_stats_resume(self, selected_year=None, force_all_years=False, filters=None):
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

            df_all = self.get_filtered_data(None, filters)
            if df_all.empty:
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

            total_presence = int(np.where(df_all['presence_flag'] > 0, 1, 0).sum())
            total_presence_current = int(np.where(current_year_data['presence_flag'] > 0, 1, 0).sum()) if len(current_year_data) > 0 else 0

            services_list = sorted(current_year_data['libelle_y'].dropna().unique().tolist()) if len(current_year_data) > 0 else []

            category_list = []
            category_columns = ['categorie', 'lib_cat', 'libelle_cat', 'libelle_categorie']
            for col in category_columns:
                if col in df_all.columns:
                    category_list = sorted(df_all[col].dropna().astype(str).unique().tolist())
                    break
            if not category_list and self.df_categories is not None:
                fallback_col = None
                for col in ['lib_cat', 'libelle', 'libelle_categorie']:
                    if col in self.df_categories.columns:
                        fallback_col = col
                        break
                if fallback_col:
                    category_list = sorted(self.df_categories[fallback_col].dropna().astype(str).unique().tolist())

            stats = {
                'total_pointages': total_presence if total_presence > 0 else int(len(df_all)),
                'total_pointages_annee_courante': total_presence_current if total_presence_current > 0 else int(len(current_year_data)),
                'employes_uniques': int(current_year_data['mat_pers'].nunique()) if len(current_year_data) > 0 else 0,
                'employes_uniques_total': int(df_all['mat_pers'].nunique()),
                'employes_actifs_annee_courante': int(current_year_data['mat_pers'].nunique()) if len(current_year_data) > 0 else 0,
                'services_actifs': len(services_list),
                'services_actifs_liste': services_list,
                'categories_actives_liste': category_list,
                'avg_daily_hours': round(df_all['duree_minutes'].mean() / 60, 2) if len(df_all) > 0 else 0,
                'avg_daily_hours_current_year': round(current_year_data['duree_minutes'].mean() / 60, 2) if len(current_year_data) > 0 else 0,
                'total_work_hours': int(df_all['duree_minutes'].sum() / 60),
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
        service_filter = request.args.get('service')
        category_filter = request.args.get('category')
        month_from = request.args.get('month_from', type=int)
        month_to = request.args.get('month_to', type=int)
        force_all_years = mode == 'all'

        if force_all_years:
            selected_year = None

        month_range = None
        if month_from or month_to:
            month_range = (month_from or 1, month_to or 12)
        filters = {
            'service': service_filter.strip() if service_filter else None,
            'category': category_filter.strip() if category_filter else None,
            'month_range': month_range
        }

        if not analyseur.charger_donnees():
            return jsonify({'error': 'Échec du chargement des données depuis la base de données'}), 500
            
        if not analyseur.nettoyer_et_preparer_donnees():
            return jsonify({'error': 'Échec du traitement des données'}), 500
        
        data = {
            'assiduite': analyseur.calculer_kpis_assiduite(selected_year, filters=filters),
            'summary': analyseur.generer_stats_resume(selected_year, force_all_years=force_all_years, filters=filters),
            'yearly_trends': analyseur.calculer_tendances_annuelles(selected_year, filters=filters),
            'employee_categories': analyseur.analyser_categories_employes(selected_year, filters=filters),
            'seasonality': analyseur.analyser_saisonnalite(selected_year, filters=filters),
            'service_efficiency': analyseur.analyser_efficacite_services(selected_year, filters=filters),
            'individual_presence': analyseur.analyser_presence_individuelle(selected_year)
        }
        
        return jsonify(data)
        
    except Exception as e:
        logging.error(f"Erreur API : {e}")
        return jsonify({'error': str(e)}), 500


@hr_analytics_bp.route('/api/employee_presence', methods=['GET'])
def get_employee_presence():
    try:
        mat_pers = request.args.get('mat_pers')
        start = request.args.get('start')
        end = request.args.get('end')

        if not mat_pers:
            return jsonify({'error': "Le paramètre 'mat_pers' est requis."}), 400

        if not analyseur.charger_donnees():
            return jsonify({'error': 'Échec du chargement des données depuis la base de données'}), 500

        if not analyseur.nettoyer_et_preparer_donnees():
            return jsonify({'error': 'Échec du traitement des données'}), 500

        result = analyseur.analyser_presence_employe(mat_pers, start, end)

        status = result.get('status')
        if status == 'not_found':
            return jsonify(result), 404

        return jsonify(result)
    except Exception as e:
        logging.error(f"Erreur API employé : {e}")
        return jsonify({'error': str(e)}), 500
