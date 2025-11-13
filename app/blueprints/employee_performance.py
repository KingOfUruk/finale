from flask import Blueprint, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import plotly
from threading import Lock
from functools import wraps

from app.config.database import get_oracle_credentials, build_sqlalchemy_url

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create Employee Performance blueprint
employee_performance_bp = Blueprint('employee_performance', __name__, template_folder='templates')
CORS(employee_performance_bp)

# Global variables for data
df_employe = None
df_pointage = None
df_categorie = None
df_service = None
employes_actuels = None
_data_lock = Lock()
_data_status = {"loaded": False, "error": None}


def init_database(force: bool = False):
    """Initialize database connection and load data"""
    global df_employe, df_pointage, df_categorie, df_service, employes_actuels
    
    with _data_lock:
        if _data_status["loaded"] and not force:
            return True
        try:
            credentials = get_oracle_credentials()
            connection_string = build_sqlalchemy_url(credentials)
            engine = create_engine(connection_string)
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1 FROM dual"))
        
            # Load data
            logging.info("Loading data from Oracle database...")
            df_employe = pd.read_sql("SELECT mat_pers, sexe, dat_nais, dat_emb, etat_civil, nbre_enf, cod_cat, code_serv FROM DIM_EMPLOYE", engine)
            df_pointage = pd.read_sql("SELECT mat_pers, date_point FROM FAIT_POINTAGE", engine)
            df_categorie = pd.read_sql("SELECT cod_cat, lib_cat FROM DIM_CATEGORIE", engine)
            df_service = pd.read_sql("SELECT code_serv, libelle FROM DIM_SERVICE", engine)
            
            # Process data
            process_data()
            logging.info("Data loaded and processed successfully")
            _data_status["loaded"] = True
            _data_status["error"] = None
            return True
            
        except Exception as e:
            logging.error(f"Database initialization failed: {str(e)}")
            _data_status["loaded"] = False
            _data_status["error"] = str(e)
            return False


def ensure_data_loaded() -> bool:
    """Ensure HR data is available before serving requests."""
    if _data_status["loaded"]:
        return True
    return init_database()


def _guard_data_ready_response():
    if ensure_data_loaded():
        return None
    return jsonify({'error': _data_status.get("error", "HR data unavailable")}), 503


def require_hr_data(func):
    """Decorator ensuring HR analytics data is ready before executing the handler."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        guard = _guard_data_ready_response()
        if guard:
            return guard
        return func(*args, **kwargs)

    return wrapper


@employee_performance_bp.before_app_request
def _warm_employee_performance_cache():
    """Prime data lazily before endpoints run."""
    ensure_data_loaded()

def process_data():
    """Process and clean the HR data"""
    global df_employe, df_pointage, df_categorie, df_service, employes_actuels
    
    # Normalize column names
    df_employe.columns = df_employe.columns.str.lower()
    df_pointage.columns = df_pointage.columns.str.lower()
    df_categorie.columns = df_categorie.columns.str.lower()
    df_service.columns = df_service.columns.str.lower()
    
    # Date processing
    df_pointage['date_point'] = pd.to_datetime(df_pointage['date_point'], errors='coerce')
    df_employe['dat_nais'] = pd.to_datetime(df_employe['dat_nais'], errors='coerce')
    df_employe['dat_emb'] = pd.to_datetime(df_employe['dat_emb'], errors='coerce')
    
    current_date = datetime.utcnow()
    
    # Normalize categorical data
    df_employe['sexe'] = df_employe['sexe'].str.upper().map({
        'M': 'Homme', 'F': 'Femme'
    }).fillna('Non spécifié')
    
    df_employe['etat_civil'] = df_employe['etat_civil'].str.upper().map({
        'M': 'Marié(e)', 'C': 'Célibataire'
    }).fillna('Non spécifié')
    
    # Calculate age and seniority
    df_employe['age'] = ((current_date - df_employe['dat_nais']).dt.days / 365.25).round().astype('Int64')
    df_employe['anciennete'] = ((current_date - df_employe['dat_emb']).dt.days / 365.25).round(1)
    
    # Process pointing data
    pointage_stats = df_pointage.groupby('mat_pers')['date_point'].agg(['min', 'max']).reset_index()
    pointage_stats.columns = ['mat_pers', 'premiere_pointage', 'derniere_apparition']
    
    df_employe = df_employe.merge(pointage_stats, on='mat_pers', how='left')
    
    # Determine employee status
    df_employe['statut'] = 'Inconnu'
    df_employe.loc[df_employe['derniere_apparition'].dt.year == 2025, 'statut'] = 'Actuel'
    
    # Calculate age at departure
    df_employe['age_a_la_sortie'] = pd.to_numeric(
        df_employe.apply(
            lambda x: ((x['derniere_apparition'] - x['dat_nais']).days / 365.25) 
            if pd.notnull(x['derniere_apparition']) and pd.notnull(x['dat_nais']) else pd.NA, 
            axis=1
        ), errors='coerce'
    ).round().astype('Int64')
    
    # Classify departure status
    condition_retraite = (
        (df_employe['derniere_apparition'].dt.year < 2025) & 
        (df_employe['age_a_la_sortie'] >= 60)
    )
    condition_demission = (
        (df_employe['derniere_apparition'].dt.year < 2025) & 
        (df_employe['age_a_la_sortie'] < 60) & 
        (df_employe['age_a_la_sortie'].notnull())
    )
    
    df_employe.loc[condition_retraite, 'statut'] = 'Retraité'
    df_employe.loc[condition_demission, 'statut'] = 'Parti/Démissionné'
    
    # Clean and enrich data
    df_employe['nbre_enf'] = df_employe['nbre_enf'].fillna(0).astype(int)
    df_employe = df_employe.merge(df_categorie, on='cod_cat', how='left')
    df_employe = df_employe.merge(df_service, on='code_serv', how='left')
    
    df_employe['annee_embauche'] = df_employe['dat_emb'].dt.year
    df_employe['annee_sortie'] = df_employe['derniere_apparition'].dt.year
    df_employe.loc[df_employe['statut'] == 'Actuel', 'annee_sortie'] = pd.NA
    
    # Get current employees
    employes_actuels = df_employe[df_employe['statut'] == 'Actuel'].copy()

@employee_performance_bp.route('/employee_performance')
def employee_performance_dashboard():
    """Render the Employee Performance dashboard"""
    data_ready = ensure_data_loaded()
    return render_template(
        'employee_performance.html',
        data_ready=data_ready,
        data_error=_data_status.get("error"),
    )

@employee_performance_bp.route('/api/overview')
@require_hr_data
def get_overview():
    """Get overview statistics"""
    try:
        total_employees = len(df_employe)
        current_employees = len(employes_actuels)
        retention_rate = current_employees / total_employees * 100 if total_employees > 0 else 0
        
        avg_age = employes_actuels['age'].mean() if len(employes_actuels) > 0 else 0
        avg_seniority = employes_actuels['anciennete'].mean() if len(employes_actuels) > 0 else 0
        
        return jsonify({
            'total_employees': total_employees,
            'current_employees': current_employees,
            'retention_rate': round(retention_rate, 1),
            'avg_age': round(avg_age, 1),
            'avg_seniority': round(avg_seniority, 1),
            'status_distribution': df_employe['statut'].value_counts().to_dict()
        })
    except Exception as e:
        logging.error(f"Error in get_overview: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/civil_status_chart')
@require_hr_data
def civil_status_chart():
    """Generate civil status pie chart"""
    try:
        status_counts = df_employe['etat_civil'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.4,
                textinfo='label+percent+value',
                textfont=dict(size=12, family="Arial Black"),
                marker=dict(
                    colors=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
                    line=dict(color='white', width=2)
                )
            )
        ])
        
        fig.update_layout(
            title={
                'text': 'Répartition de l\'État Civil - Tous Employés',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            annotations=[
                dict(
                    text=f'<b>Total<br>{len(df_employe):,}</b>', 
                    x=0.5, y=0.5, 
                    font_size=14, 
                    showarrow=False
                )
            ],
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON)
    except Exception as e:
        logging.error(f"Error in civil_status_chart: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/age_pyramid')
@require_hr_data
def age_pyramid():
    """Generate age pyramid"""
    try:
        if len(employes_actuels) == 0:
            return jsonify({'error': 'No current employees data'}), 400
            
        employes_avec_age = employes_actuels[employes_actuels['age'].notna()].copy()
        
        if len(employes_avec_age) == 0:
            return jsonify({'error': 'No age data available'}), 400
        
        # Create age groups
        bins = list(range(20, 70, 5))
        labels = [f'{i}-{i+4}' for i in bins[:-1]] + ['65+']
        
        employes_avec_age['groupe_age'] = pd.cut(
            employes_avec_age['age'], 
            bins=bins + [100], 
            labels=labels, 
            right=False
        )
        
        # Count by age group and gender
        pyramide_data = employes_avec_age.groupby(['groupe_age', 'sexe']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        if 'Homme' in pyramide_data.columns:
            fig.add_trace(go.Bar(
                name='Hommes',
                y=pyramide_data.index,
                x=-pyramide_data['Homme'],
                orientation='h',
                marker_color='#2E86AB',
                text=pyramide_data['Homme'].values,
                textposition='inside'
            ))
        
        if 'Femme' in pyramide_data.columns:
            fig.add_trace(go.Bar(
                name='Femmes',
                y=pyramide_data.index,
                x=pyramide_data['Femme'],
                orientation='h',
                marker_color='#A23B72',
                text=pyramide_data['Femme'].values,
                textposition='inside'
            ))
        
        fig.update_layout(
            title={
                'text': 'Pyramide des Âges - Employés Actuels',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            xaxis_title='Nombre d\'Employés',
            yaxis_title='Groupe d\'Âge',
            barmode='overlay',
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=500
        )
        
        # Customize X-axis to show absolute values
        max_val = max(pyramide_data.max()) if len(pyramide_data.columns) > 0 else 10
        fig.update_xaxes(
            tickvals=list(range(-max_val, max_val+5, 5)),
            ticktext=[str(abs(x)) for x in range(-max_val, max_val+5, 5)]
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON)
    except Exception as e:
        logging.error(f"Error in age_pyramid: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/hiring_trend')
@require_hr_data
def hiring_trend():
    """Generate hiring trend chart"""
    try:
        hiring_by_year = df_employe['annee_embauche'].value_counts().sort_index()
        
        if hiring_by_year.empty:
            return jsonify({'error': 'No hiring data available'}), 400
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hiring_by_year.index,
            y=hiring_by_year.values,
            mode='lines+markers',
            name='Embauches',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color='#A23B72'),
            fill='tonexty',
            fillcolor='rgba(46, 134, 171, 0.1)'
        ))
        
        # Add trend line
        z = np.polyfit(hiring_by_year.index, hiring_by_year.values, 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=hiring_by_year.index,
            y=p(hiring_by_year.index),
            mode='lines',
            name='Tendance',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title={
                'text': 'Évolution Historique des Embauches',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            xaxis_title='Année',
            yaxis_title='Nombre d\'Embauches',
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400,
            hovermode='x unified'
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON)
    except Exception as e:
        logging.error(f"Error in hiring_trend: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/service_category')
@require_hr_data
def service_category():
    """Generate service/category distribution chart"""
    try:
        if len(employes_actuels) == 0:
            return jsonify({'error': 'No current employees data'}), 400
            
        employes_clean = employes_actuels[
            employes_actuels['lib_cat'].notna() & 
            employes_actuels['libelle'].notna()
        ].copy()
        
        if len(employes_clean) == 0:
            return jsonify({'error': 'No category/service data available'}), 400
        
        # Create cross table
        cross_tab = pd.crosstab(
            employes_clean['lib_cat'], 
            employes_clean['libelle'], 
            margins=False
        )
        
        cross_tab['Total'] = cross_tab.sum(axis=1)
        cross_tab = cross_tab.sort_values('Total', ascending=True)
        cross_tab = cross_tab.drop('Total', axis=1)
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3[:len(cross_tab.columns)]
        
        for i, service in enumerate(cross_tab.columns):
            fig.add_trace(go.Bar(
                name=service,
                y=cross_tab.index,
                x=cross_tab[service],
                orientation='h',
                marker_color=colors[i % len(colors)],
                text=cross_tab[service].values,
                textposition='inside'
            ))
        
        fig.update_layout(
            title={
                'text': 'Répartition par Catégorie et Service - Employés Actuels',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            xaxis_title='Nombre d\'Employés',
            yaxis_title='Catégorie Professionnelle',
            barmode='stack',
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=max(400, len(cross_tab) * 60)
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON)
    except Exception as e:
        logging.error(f"Error in service_category: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/gender_distribution')
@require_hr_data
def gender_distribution():
    """Get gender distribution data"""
    try:
        if len(employes_actuels) == 0:
            return jsonify({'error': 'No current employees data'}), 400
            
        gender_counts = employes_actuels['sexe'].value_counts()
        gender_percentages = (gender_counts / len(employes_actuels) * 100).round(1)
        
        return jsonify({
            'counts': gender_counts.to_dict(),
            'percentages': gender_percentages.to_dict(),
            'total': len(employes_actuels)
        })
    except Exception as e:
        logging.error(f"Error in gender_distribution: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/children_distribution')
@require_hr_data
def children_distribution():
    """Generate children distribution chart"""
    try:
        if len(employes_actuels) == 0:
            return jsonify({'error': 'No current employees data'}), 400
        
        children_counts = employes_actuels['nbre_enf'].value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=children_counts.index,
                y=children_counts.values,
                marker_color='#2E86AB',
                text=children_counts.values,
                textposition='outside',
                name='Nombre d\'employés'
            )
        ])
        
        fig.update_layout(
            title={
                'text': 'Nombre d\'Enfants - Employés Actuels',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            xaxis_title='Nombre d\'Enfants',
            yaxis_title='Nombre d\'Employés',
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON)
    except Exception as e:
        logging.error(f"Error in children_distribution: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/seniority_analysis')
@require_hr_data
def seniority_analysis():
    """Get seniority analysis data"""
    try:
        if len(employes_actuels) == 0:
            return jsonify({'error': 'No current employees data'}), 400
            
        # Create seniority categories
        bins = [0, 2, 5, 10, 15, 50]
        labels = ['0-2 ans', '2-5 ans', '5-10 ans', '10-15 ans', '15+ ans']
        
        employes_actuels_seniority = employes_actuels[employes_actuels['anciennete'].notna()].copy()
        employes_actuels_seniority['categorie_anciennete'] = pd.cut(
            employes_actuels_seniority['anciennete'],
            bins=bins,
            labels=labels,
            right=False
        )
        
        seniority_counts = employes_actuels_seniority['categorie_anciennete'].value_counts()
        seniority_percentages = (seniority_counts / len(employes_actuels_seniority) * 100).round(1)
        
        return jsonify({
            'counts': seniority_counts.to_dict(),
            'percentages': seniority_percentages.to_dict(),
            'avg_seniority': round(employes_actuels['anciennete'].mean(), 1),
            'median_seniority': round(employes_actuels['anciennete'].median(), 1)
        })
    except Exception as e:
        logging.error(f"Error in seniority_analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/seniority_by_service')
@require_hr_data
def seniority_by_service():
    """Generate seniority distribution by service chart"""
    try:
        if len(employes_actuels) == 0:
            return jsonify({'error': 'No current employees data'}), 400
            
        # Create seniority categories
        bins = [0, 2, 5, 10, 15, 50]
        labels = ['0-2 ans', '2-5 ans', '5-10 ans', '10-15 ans', '15+ ans']
        
        employes_clean = employes_actuels[
            employes_actuels['anciennete'].notna() & 
            employes_actuels['libelle'].notna()
        ].copy()
        
        if len(employes_clean) == 0:
            return jsonify({'error': 'No seniority/service data available'}), 400
        
        employes_clean['categorie_anciennete'] = pd.cut(
            employes_clean['anciennete'],
            bins=bins,
            labels=labels,
            right=False
        )
        
        # Create cross table
        seniority_service = pd.crosstab(
            employes_clean['libelle'],
            employes_clean['categorie_anciennete']
        ).fillna(0)
        
        # Calculate percentages
        seniority_service_pct = seniority_service.div(seniority_service.sum(axis=1), axis=0) * 100
        
        fig = go.Figure()
        
        colors = ['#E8F4FD', '#B3D9F7', '#7FB8E5', '#4A90D3', '#1565C0']
        
        for i, category in enumerate(seniority_service_pct.columns):
            fig.add_trace(go.Bar(
                name=category,
                x=seniority_service_pct.index,
                y=seniority_service_pct[category],
                marker_color=colors[i],
                text=[f'{val:.0f}%' if val > 5 else '' for val in seniority_service_pct[category]],
                textposition='inside'
            ))
        
        fig.update_layout(
            title={
                'text': 'Répartition de l\'Ancienneté par Service (%) - Employés Actuels',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            xaxis_title='Service',
            yaxis_title='Pourcentage (%)',
            barmode='stack',
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=500
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(range=[0, 100])
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON)
    except Exception as e:
        logging.error(f"Error in seniority_by_service: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/seniority_by_gender')
@require_hr_data
def seniority_by_gender():
    """Generate seniority by service and gender chart"""
    try:
        if len(employes_actuels) == 0:
            return jsonify({'error': 'No current employees data'}), 400
            
        employes_clean = employes_actuels[
            employes_actuels['anciennete'].notna() & 
            employes_actuels['libelle'].notna() &
            employes_actuels['sexe'].notna()
        ].copy()
        
        if len(employes_clean) == 0:
            return jsonify({'error': 'No complete data available'}), 400
        
        # Calculate average seniority by service and gender
        seniority_service_gender = employes_clean.groupby(['libelle', 'sexe'])['anciennete'].mean().unstack(fill_value=0)
        seniority_service_gender = seniority_service_gender[
            (seniority_service_gender.sum(axis=1) > 0)
        ].sort_values(seniority_service_gender.columns[0] if len(seniority_service_gender.columns) > 0 else 'libelle', ascending=True)
        
        fig = go.Figure()
        
        colors = {'Homme': '#2E86AB', 'Femme': '#A23B72', 'Non spécifié': '#95A5A6'}
        
        for sexe in seniority_service_gender.columns:
            if sexe != 'Non spécifié':
                fig.add_trace(go.Bar(
                    name=sexe,
                    y=seniority_service_gender.index,
                    x=seniority_service_gender[sexe],
                    orientation='h',
                    marker_color=colors.get(sexe, '#95A5A6'),
                    text=[f'{val:.1f}' if val > 0 else '' for val in seniority_service_gender[sexe]],
                    textposition='inside'
                ))
        
        fig.update_layout(
            title={
                'text': 'Ancienneté Moyenne par Service et Genre - Employés Actuels',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            xaxis_title='Ancienneté Moyenne (années)',
            yaxis_title='Service',
            barmode='group',
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=max(400, len(seniority_service_gender) * 50)
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON)
    except Exception as e:
        logging.error(f"Error in seniority_by_gender: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/gender_evolution')
@require_hr_data
def gender_evolution():
    """Generate gender diversity evolution over time"""
    try:
        # Group by hiring year and gender
        gender_evolution = df_employe.groupby(['annee_embauche', 'sexe']).size().unstack(fill_value=0)
        
        if gender_evolution.empty:
            return jsonify({'error': 'No hiring evolution data available'}), 400
        
        fig = go.Figure()
        
        colors = {'Homme': '#2E86AB', 'Femme': '#A23B72', 'Non spécifié': '#95A5A6'}
        
        for sexe in gender_evolution.columns:
            if sexe != 'Non spécifié':
                fig.add_trace(go.Scatter(
                    x=gender_evolution.index,
                    y=gender_evolution[sexe],
                    name=sexe,
                    mode='lines+markers',
                    stackgroup='one',
                    line=dict(width=2),
                    marker=dict(size=6),
                    fillcolor=colors.get(sexe, '#95A5A6')
                ))
        
        fig.update_layout(
            title={
                'text': 'Évolution de la Diversité de Genre dans les Embauches',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            xaxis_title='Année d\'Embauche',
            yaxis_title='Nombre d\'Embauches',
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400,
            hovermode='x unified'
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON)
    except Exception as e:
        logging.error(f"Error in gender_evolution: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/turnover_analysis')
@require_hr_data
def turnover_analysis():
    """Generate turnover analysis chart"""
    try:
        # Departures by year
        departures_by_year = df_employe[
            (df_employe['annee_sortie'].notna()) & 
            (df_employe['annee_sortie'] < 2025)
        ]['annee_sortie'].value_counts().sort_index()
        
        hiring_by_year = df_employe['annee_embauche'].value_counts().sort_index()
        
        # Calculate net turnover by year
        common_years = set(departures_by_year.index) & set(hiring_by_year.index)
        turnover_data = []
        
        for year in sorted(common_years):
            hires = hiring_by_year.get(year, 0)
            departures = departures_by_year.get(year, 0)
            net = hires - departures
            turnover_data.append({
                'year': year,
                'hires': hires,
                'departures': departures,
                'net': net
            })
        
        if not turnover_data:
            return jsonify({'error': 'No turnover data available'}), 400
        
        df_turnover = pd.DataFrame(turnover_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Embauches',
            x=df_turnover['year'],
            y=df_turnover['hires'],
            marker_color='#2E86AB',
            text=df_turnover['hires'],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Départs',
            x=df_turnover['year'],
            y=-df_turnover['departures'],
            marker_color='#C73E1D',
            text=df_turnover['departures'],
            textposition='outside'
        ))
        
        fig.add_trace(go.Scatter(
            name='Solde Net',
            x=df_turnover['year'],
            y=df_turnover['net'],
            mode='lines+markers',
            line=dict(color='#F18F01', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title={
                'text': 'Analyse des Embauches et Départs par Année',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            xaxis_title='Année',
            yaxis=dict(title='Embauches / Départs'),
            yaxis2=dict(title='Solde Net', side='right', overlaying='y'),
            barmode='relative',
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400,
            hovermode='x unified'
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON)
    except Exception as e:
        logging.error(f"Error in turnover_analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/age_distribution_by_gender')
@require_hr_data
def age_distribution_by_gender():
    """Generate age distribution by gender chart"""
    try:
        if len(employes_actuels) == 0:
            return jsonify({'error': 'No current employees data'}), 400
            
        employes_with_age = employes_actuels[employes_actuels['age'].notna()].copy()
        
        if len(employes_with_age) == 0:
            return jsonify({'error': 'No age data available'}), 400
        
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=['Distribution des Âges par Genre', 'Box Plot des Âges'])
        
        colors = {'Homme': '#2E86AB', 'Femme': '#A23B72', 'Non spécifié': '#95A5A6'}
        
        # Histogram
        for genre in employes_with_age['sexe'].unique():
            if genre != 'Non spécifié':
                data = employes_with_age[employes_with_age['sexe'] == genre]['age'].dropna()
                fig.add_trace(go.Histogram(
                    x=data,
                    name=f'{genre} (n={len(data)})',
                    opacity=0.7,
                    marker_color=colors.get(genre, '#95A5A6'),
                    nbinsx=15
                ), row=1, col=1)
        
        # Box plot
        for genre in employes_with_age['sexe'].unique():
            if genre != 'Non spécifié':
                data = employes_with_age[employes_with_age['sexe'] == genre]['age'].dropna()
                fig.add_trace(go.Box(
                    y=data,
                    name=genre,
                    marker_color=colors.get(genre, '#95A5A6'),
                    showlegend=False
                ), row=1, col=2)
        
        fig.update_layout(
            title={
                'text': 'Distribution des Âges par Genre - Employés Actuels',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=500
        )
        
        fig.update_xaxes(title_text="Âge (années)", row=1, col=1)
        fig.update_yaxes(title_text="Nombre d'Employés", row=1, col=1)
        fig.update_yaxes(title_text="Âge (années)", row=1, col=2)
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON)
    except Exception as e:
        logging.error(f"Error in age_distribution_by_gender: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/data_completeness')
@require_hr_data
def data_completeness():
    """Generate data completeness heatmap"""
    try:
        def calculate_completeness(df, columns):
            completeness = {}
            for status in df['statut'].unique():
                df_status = df[df['statut'] == status]
                completeness[status] = {}
                for col in columns:
                    if col in df_status.columns:
                        pct_complete = (1 - df_status[col].isnull().mean()) * 100
                        completeness[status][col] = pct_complete
            return pd.DataFrame(completeness).T
        
        important_columns = ['dat_nais', 'dat_emb', 'derniere_apparition', 'lib_cat', 'libelle']
        df_completeness = calculate_completeness(df_employe, important_columns)
        
        if df_completeness.empty:
            return jsonify({'error': 'No completeness data available'}), 400
        
        fig = go.Figure(data=go.Heatmap(
            z=df_completeness.values,
            x=df_completeness.columns,
            y=df_completeness.index,
            colorscale='RdYlGn',
            text=df_completeness.round(1).astype(str) + '%',
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Complétude (%)")
        ))
        
        fig.update_layout(
            title={
                'text': 'Complétude des Données par Statut d\'Employé',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            xaxis_title='Colonnes',
            yaxis_title='Statut des Employés',
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON)
    except Exception as e:
        logging.error(f"Error in data_completeness: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/top_services')
@require_hr_data
def top_services():
    """Get top 5 services by employee count"""
    try:
        if len(employes_actuels) == 0:
            return jsonify({'error': 'No current employees data'}), 400
            
        # Get top services
        services_clean = employes_actuels[employes_actuels['libelle'].notna()]
        
        if len(services_clean) == 0:
            return jsonify({'error': 'No service data available'}), 400
        
        top_services_data = services_clean['libelle'].value_counts().head(5)
        total_employees = len(services_clean)
        
        services_info = []
        for service, count in top_services_data.items():
            percentage = round((count / total_employees * 100), 1)
            services_info.append({
                'service': service,
                'count': int(count),
                'percentage': float(percentage)
            })
        
        return jsonify({
            'services': services_info,
            'total_employees': total_employees
        })
        
    except Exception as e:
        logging.error(f"Error in top_services: {str(e)}")
        return jsonify({'error': str(e)}), 500

@employee_performance_bp.route('/api/refresh_data')
def refresh_data():
    """Refresh data from database"""
    try:
        success = init_database(force=True)
        if success:
            return jsonify({'success': True, 'message': 'Data refreshed successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to refresh data'}), 500
    except Exception as e:
        logging.error(f"Error in refresh_data: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

