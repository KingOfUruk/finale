from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np
import sqlalchemy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os
from app.config.database import get_oracle_credentials
from app.oracle_helpers import make_dsn

warnings.filterwarnings('ignore')

# Enhanced plotting style
sns.set_style('whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

prediction_bp = Blueprint('prediction', __name__, url_prefix='/api/predictions')

class EnhancedPayrollPredictiveModel:
    """
    Enhanced predictive model with better features and comprehensive visualizations
    """
    
    def __init__(self):
        self.db_config = get_oracle_credentials()
        
        self.engine = None
        self.salary_model = None
        self.advance_classifier = None
        self.advance_regressor = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.salary_feature_names = []  # Store salary model features
        self.advance_feature_names = []  # Store advance model features
        self.training_metadata = {}
        self.department_stats = {}
        self.category_stats = {}
        self.df_salary = None
        self.df_advance = None
        self.initialization_error = None
        
        self.connect_db()
    
    def connect_db(self):
        """Connect to Oracle database"""
        try:
            port_value = int(self.db_config['port']) if str(self.db_config['port']).isdigit() else self.db_config['port']
            dsn = make_dsn(
                self.db_config['host'],
                port_value,
                self.db_config['service_name']
            )
            driver = self.db_config.get('driver', 'oracle+oracledb')
            connection_string = f"{driver}://{self.db_config['username']}:{self.db_config['password']}@{dsn}"
            self.engine = sqlalchemy.create_engine(connection_string)
            
            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1 FROM dual"))
            
            print("✓ Database connection established")
            return True
        except Exception as e:
            self.initialization_error = str(e)
            print(f"✗ Database connection failed: {e}")
            return False
    
    def load_data(self, year=None):
        """Load data from database"""
        if not self.engine:
            raise RuntimeError(self.initialization_error or "No database connection")
        
        year_filter = f"AND dt.annee = {year}" if year else "AND dt.annee >= 2018"
        
        query = f"""
        SELECT 
            fr.id_fact, fr.mat_pers, fr.id_temps, fr.cod_niv, fr.montant, fr.type_paie, fr.source,
            emp.sexe, emp.dat_nais, emp.dat_emb, emp.etat_civil, emp.nbre_enf, emp.cod_cat,
            emp.code_serv, emp.age, emp.anciennete, emp.statut, emp.lib_cat, emp.libelle as service_libelle,
            dt.date_jour, dt.annee, dt.trimestre, dt.mois, dt.est_weekend
        FROM FAIT_remuneration fr
        LEFT JOIN DIM_EMPLOYEe_nouveau emp ON fr.mat_pers = emp.mat_pers
        LEFT JOIN DIM_TEMPS_nouveau dt ON fr.id_temps = dt.id_temps
        WHERE 1=1 {year_filter}
        """
        
        print(f"\nLoading data...")
        df = pd.read_sql(query, self.engine)
        
        # Clean and type conversion
        numeric_cols = ['montant', 'age', 'anciennete', 'nbre_enf']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        categorical_cols = ['cod_cat', 'code_serv', 'sexe', 'etat_civil', 'lib_cat']
        for col in categorical_cols:
            df[col] = df[col].astype(str).fillna('Unknown')
        
        print(f"✓ Loaded {len(df):,} records | {df['annee'].min()}-{df['annee'].max()} | {df['mat_pers'].nunique():,} employees")
        return df
    
    def categorize_payments(self, df):
        """Enhanced payment categorization"""
        df = df.copy()
        
        def categorize(row):
            source = str(row.get('source', '')).lower()
            montant = float(row['montant'])
            
            if 'snet' in source or 'sbrut' in source:
                return 'Salaire'
            elif 'prime' in source or 'pprime' in source or 'nprime' in source:
                return 'Prime'
            elif 'ava' in source or ('pres' in source and montant < 0):
                return 'Avance'
            elif montant < 0:
                return 'Retenue'
            elif montant > 0:
                return 'Autre_Remuneration'
            return 'Autre'
        
        df['payment_category'] = df.apply(categorize, axis=1)
        return df
    
    def engineer_features(self, df):
        """Advanced feature engineering"""
        df = df.copy()
        
        # Tenure bands
        df['tenure_band'] = pd.cut(df['anciennete'], 
                                    bins=[-np.inf, 1, 3, 5, 10, np.inf],
                                    labels=['<1yr', '1-3yrs', '3-5yrs', '5-10yrs', '10+yrs'])
        
        # Age bands
        df['age_band'] = pd.cut(df['age'], 
                                bins=[0, 25, 35, 45, 55, 100],
                                labels=['<25', '25-35', '35-45', '45-55', '55+'])
        
        # Family status
        df['has_children'] = (df['nbre_enf'] > 0).astype(int)
        df['family_size'] = df['nbre_enf'] + 1  # Include self
        
        # Career stage
        df['career_stage'] = 'Junior'
        df.loc[(df['anciennete'] >= 3) & (df['anciennete'] < 10), 'career_stage'] = 'Mid-level'
        df.loc[df['anciennete'] >= 10, 'career_stage'] = 'Senior'
        
        return df
    
    def calculate_department_benchmarks(self, df):
        """Calculate department and category statistics for benchmarking"""
        if 'payment_category' not in df.columns:
            raise ValueError("DataFrame missing 'payment_category' column. Run categorize_payments first.")
        df_salary = df[df['payment_category'] == 'Salaire'].copy()
        
        # Department statistics
        dept_stats = df_salary.groupby('code_serv').agg({
            'montant': ['mean', 'median', 'std', 'count']
        }).round(2)
        dept_stats.columns = ['avg_salary', 'median_salary', 'std_salary', 'count']
        self.department_stats = dept_stats.to_dict('index')
        
        # Category statistics
        cat_stats = df_salary.groupby('cod_cat').agg({
            'montant': ['mean', 'median', 'std', 'count']
        }).round(2)
        cat_stats.columns = ['avg_salary', 'median_salary', 'std_salary', 'count']
        self.category_stats = cat_stats.to_dict('index')
        
        print(f"✓ Calculated benchmarks for {len(dept_stats)} departments and {len(cat_stats)} categories")
    
    def prepare_salary_data(self, df):
        """Enhanced salary data preparation"""
        print("\n" + "="*70)
        print("PREPARING SALARY DATA WITH ENHANCED FEATURES")
        print("="*70)
        
        # Get salary data
        df_salary = df[df['payment_category'] == 'Salaire'].copy()
        
        # Aggregate by employee and year
        agg_dict = {
            'montant': 'sum',
            'age': 'first',
            'anciennete': 'first',
            'cod_cat': 'first',
            'code_serv': 'first',
            'sexe': 'first',
            'nbre_enf': 'first',
            'etat_civil': 'first',
            'lib_cat': 'first',
            'tenure_band': 'first',
            'age_band': 'first',
            'has_children': 'first',
            'career_stage': 'first'
        }
        
        df_agg = df_salary.groupby(['mat_pers', 'annee']).agg(agg_dict).reset_index()
        df_agg.columns = ['mat_pers', 'annee', 'salary_total', 'age', 'anciennete', 
                         'cod_cat', 'code_serv', 'sexe', 'nbre_enf', 'etat_civil', 'lib_cat',
                         'tenure_band', 'age_band', 'has_children', 'career_stage']
        
        # Sort and calculate growth
        df_agg = df_agg.sort_values(['mat_pers', 'annee'])
        
        # Previous salary
        df_agg['prev_salary'] = df_agg.groupby('mat_pers')['salary_total'].shift(1)
        df_agg['salary_2yrs_ago'] = df_agg.groupby('mat_pers')['salary_total'].shift(2)
        
        # Growth metrics
        df_agg['salary_growth'] = df_agg['salary_total'] - df_agg['prev_salary']
        df_agg['salary_growth_pct'] = (df_agg['salary_growth'] / df_agg['prev_salary']) * 100
        
        # Cap unrealistic growth (likely data errors)
        df_agg['salary_growth_pct'] = df_agg['salary_growth_pct'].clip(-50, 100)
        
        # Add department benchmark features
        df_agg['dept_avg_salary'] = df_agg['code_serv'].map(
            lambda x: self.department_stats.get(x, {}).get('avg_salary', 0)
        )
        df_agg['salary_vs_dept_avg'] = (df_agg['salary_total'] / df_agg['dept_avg_salary']) * 100
        
        # Years in company
        df_agg['years_employed'] = df_agg['anciennete'].astype(int)
        
        # Remove first year for each employee
        df_clean = df_agg.dropna(subset=['prev_salary'])
        
        # Remove outliers (extreme growth)
        Q1 = df_clean['salary_growth_pct'].quantile(0.05)
        Q3 = df_clean['salary_growth_pct'].quantile(0.95)
        df_clean = df_clean[(df_clean['salary_growth_pct'] >= Q1) & 
                           (df_clean['salary_growth_pct'] <= Q3)]
        
        print(f"✓ Prepared {len(df_clean):,} salary records")
        print(f"  Employees: {df_clean['mat_pers'].nunique():,}")
        print(f"  Avg salary: {df_clean['salary_total'].mean():,.2f} TND")
        print(f"  Avg growth: {df_clean['salary_growth_pct'].mean():.2f}%")
        print(f"  Growth range: {df_clean['salary_growth_pct'].min():.1f}% to {df_clean['salary_growth_pct'].max():.1f}%")
        
        return df_clean
    
    def prepare_advance_data(self, df):
        """Enhanced advance data preparation"""
        print("\n" + "="*70)
        print("PREPARING ADVANCE DATA WITH ENHANCED FEATURES")
        print("="*70)
        
        # Get advances
        df_advance = df[df['payment_category'] == 'Avance'].copy()
        df_advance['montant_abs'] = df_advance['montant'].abs()
        
        advance_summary = df_advance.groupby(['mat_pers', 'annee']).agg({
            'montant_abs': ['sum', 'mean', 'count'],
            'mois': lambda x: x.nunique()
        }).reset_index()
        advance_summary.columns = ['mat_pers', 'annee', 'advance_total', 
                                   'advance_avg', 'advance_count', 'months_with_advance']
        
        # Get all employees
        agg_dict = {
            'age': 'first',
            'anciennete': 'first',
            'cod_cat': 'first',
            'code_serv': 'first',
            'sexe': 'first',
            'nbre_enf': 'first',
            'etat_civil': 'first',
            'lib_cat': 'first',
            'tenure_band': 'first',
            'has_children': 'first',
            'career_stage': 'first'
        }
        
        df_all = df.groupby(['mat_pers', 'annee']).agg(agg_dict).reset_index()
        
        # Get salaries
        df_salary = df[df['payment_category'] == 'Salaire'].groupby(['mat_pers', 'annee'])['montant'].sum().reset_index()
        df_salary.columns = ['mat_pers', 'annee', 'salary']
        
        # Merge
        df_all = df_all.merge(df_salary, on=['mat_pers', 'annee'], how='left')
        df_all = df_all.merge(advance_summary, on=['mat_pers', 'annee'], how='left')
        
        # Fill NaN
        advance_cols = ['advance_total', 'advance_avg', 'advance_count', 'months_with_advance']
        for col in advance_cols:
            df_all[col] = df_all[col].fillna(0)
        
        # Target variables
        df_all['has_advance'] = (df_all['advance_total'] > 0).astype(int)
        df_all['advance_ratio'] = np.where(df_all['salary'] > 0,
                                           (df_all['advance_total'] / df_all['salary']) * 100, 0)
        df_all['advance_frequency'] = df_all['months_with_advance'] / 12  # Proportion of year
        
        # Risk categories
        df_all['advance_risk'] = 'Low'
        df_all.loc[df_all['advance_ratio'] > 10, 'advance_risk'] = 'Medium'
        df_all.loc[df_all['advance_ratio'] > 30, 'advance_risk'] = 'High'
        
        print(f"✓ Prepared {len(df_all):,} employee-year records")
        print(f"  With advances: {df_all['has_advance'].sum():,} ({df_all['has_advance'].mean()*100:.1f}%)")
        print(f"  Avg advance: {df_all[df_all['has_advance']==1]['advance_total'].mean():.2f} TND")
        print(f"  High risk: {(df_all['advance_risk']=='High').sum():,} employees")
        
        return df_all
    
    def fit_label_encoders(self, df):
        """Fit label encoders"""
        categorical_cols = ['cod_cat', 'code_serv', 'sexe', 'etat_civil', 
                           'tenure_band', 'age_band', 'career_stage']
        
        for col in categorical_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                valid_data = df[col].astype(str).fillna('Unknown')
                self.label_encoders[col].fit(valid_data)
    
    def train_salary_model(self, df_salary):
        """Train enhanced salary prediction model"""
        print("\n" + "="*70)
        print("TRAINING SALARY PREDICTION MODEL")
        print("="*70)
        
        # Enhanced features
        feature_cols = [
            'prev_salary', 'age', 'anciennete', 'nbre_enf',
            'cod_cat', 'code_serv', 'sexe', 'etat_civil',
            'has_children', 'years_employed', 'dept_avg_salary',
            'salary_vs_dept_avg', 'tenure_band', 'career_stage'
        ]
        
        # Remove features not in dataframe
        feature_cols = [col for col in feature_cols if col in df_salary.columns]
        
        X = df_salary[feature_cols].copy()
        y = df_salary['salary_total']
        
        # Encode categorical
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if col in self.label_encoders:
                X[col] = self.label_encoders[col].transform(X[col].astype(str).fillna('Unknown'))
        
        X = X.fillna(X.median())
        self.salary_feature_names = X.columns.tolist()  # Store salary model features
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Train models with hyperparameter tuning
        models = {
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=150, max_depth=7, learning_rate=0.05,
                min_samples_split=5, random_state=42
            )
        }
        
        results = {}
        print("\nModel Comparison:")
        print("-" * 70)
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'CV_R2_mean': cv_scores.mean(),
                'CV_R2_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"\n{name}:")
            print(f"  MAE: {mae:,.2f} TND | RMSE: {rmse:,.2f} TND")
            print(f"  R²: {r2:.4f} | MAPE: {mape:.2f}%")
            print(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        # Select best model (by R²)
        best_model_name = max(results, key=lambda k: results[k]['R2'])
        self.salary_model = results[best_model_name]['model']
        
        # Store metrics in training_metadata
        self.training_metadata['salary'] = {
            'metrics': results[best_model_name],
            'n_samples': len(X_train)
        }
        
        print("\n" + "="*70)
        print(f"BEST MODEL: {best_model_name} (R²={results[best_model_name]['R2']:.4f})")
        print("="*70)
        
        # Visualizations
        self._plot_salary_model_diagnostics(results, best_model_name, X_test, y_test, y_train)
        
        return results
    
    def _plot_salary_model_diagnostics(self, results, best_model_name, X_test, y_test, y_train):
        """Create comprehensive diagnostic plots"""
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Model Comparison
        ax1 = plt.subplot(3, 3, 1)
        model_names = list(results.keys())
        r2_scores = [results[m]['R2'] for m in model_names]
        colors = ['#2ecc71' if m == best_model_name else '#3498db' for m in model_names]
        ax1.barh(model_names, r2_scores, color=colors)
        ax1.set_xlabel('R² Score')
        ax1.set_title('Model Performance Comparison', fontweight='bold')
        ax1.set_xlim(0, 1)
        
        # 2. MAE Comparison
        ax2 = plt.subplot(3, 3, 2)
        mae_scores = [results[m]['MAE'] for m in model_names]
        ax2.barh(model_names, mae_scores, color=colors)
        ax2.set_xlabel('MAE (TND)')
        ax2.set_title('Mean Absolute Error', fontweight='bold')
        
        # 3. Actual vs Predicted
        ax3 = plt.subplot(3, 3, 3)
        y_pred_best = results[best_model_name]['predictions']
        ax3.scatter(y_test, y_pred_best, alpha=0.6, s=50)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax3.set_xlabel('Actual Salary (TND)')
        ax3.set_ylabel('Predicted Salary (TND)')
        ax3.set_title(f'{best_model_name}: Actual vs Predicted', fontweight='bold')
        ax3.legend()
        
        # 4. Residuals
        ax4 = plt.subplot(3, 3, 4)
        residuals = y_test - y_pred_best
        ax4.scatter(y_pred_best, residuals, alpha=0.6, s=50)
        ax4.axhline(y=0, color='r', linestyle='--', lw=2)
        ax4.set_xlabel('Predicted Salary (TND)')
        ax4.set_ylabel('Residuals (TND)')
        ax4.set_title('Residual Plot', fontweight='bold')
        
        # 5. Residual Distribution
        ax5 = plt.subplot(3, 3, 5)
        ax5.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax5.axvline(x=0, color='r', linestyle='--', lw=2)
        ax5.set_xlabel('Residuals (TND)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Residual Distribution', fontweight='bold')
        
        # 6. Feature Importance
        ax6 = plt.subplot(3, 3, 6)
        if hasattr(self.salary_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.salary_feature_names,
                'importance': self.salary_model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)
            
            ax6.barh(importance_df['feature'], importance_df['importance'])
            ax6.set_xlabel('Importance')
            ax6.set_title('Top 10 Feature Importance', fontweight='bold')
        elif hasattr(self.salary_model, 'coef_'):
            coef_df = pd.DataFrame({
                'feature': self.salary_feature_names,
                'coefficient': np.abs(self.salary_model.coef_)
            }).sort_values('coefficient', ascending=True).tail(10)
            
            ax6.barh(coef_df['feature'], coef_df['coefficient'])
            ax6.set_xlabel('Absolute Coefficient')
            ax6.set_title('Top 10 Feature Coefficients', fontweight='bold')
        
        # 7. Salary Distribution
        ax7 = plt.subplot(3, 3, 7)
        ax7.hist(y_train, bins=40, alpha=0.7, label='Training', edgecolor='black')
        ax7.hist(y_test, bins=40, alpha=0.7, label='Test', edgecolor='black')
        ax7.set_xlabel('Salary (TND)')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Salary Distribution', fontweight='bold')
        ax7.legend()
        
        # 8. Error by Salary Range
        ax8 = plt.subplot(3, 3, 8)
        salary_bins = pd.qcut(y_test, q=5, duplicates='drop')
        error_by_bin = pd.DataFrame({
            'salary_bin': salary_bins,
            'abs_error': np.abs(residuals)
        }).groupby('salary_bin')['abs_error'].mean()
        
        ax8.bar(range(len(error_by_bin)), error_by_bin.values)
        ax8.set_xlabel('Salary Quintile (Low to High)')
        ax8.set_ylabel('Mean Absolute Error (TND)')
        ax8.set_title('Error by Salary Range', fontweight='bold')
        ax8.set_xticks(range(len(error_by_bin)))
        ax8.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        # 9. Prediction Confidence
        ax9 = plt.subplot(3, 3, 9)
        error_pct = (np.abs(residuals) / y_test) * 100
        ax9.scatter(y_test, error_pct, alpha=0.6, s=50)
        ax9.axhline(y=10, color='orange', linestyle='--', label='10% Error', lw=2)
        ax9.axhline(y=20, color='red', linestyle='--', label='20% Error', lw=2)
        ax9.set_xlabel('Actual Salary (TND)')
        ax9.set_ylabel('Error (%)')
        ax9.set_title('Prediction Error %', fontweight='bold')
        ax9.legend()
        ax9.set_ylim(0, min(error_pct.max(), 50))
        
        plt.tight_layout()
        plt.savefig('salary_model_diagnostics.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: salary_model_diagnostics.png")
        plt.close()
    
    def train_advance_models(self, df_advance):
        """Train advance prediction models with visualizations"""
        print("\n" + "="*70)
        print("TRAINING ADVANCE PREDICTION MODELS")
        print("="*70)
        
        feature_cols = [
            'salary', 'age', 'anciennete', 'nbre_enf',
            'cod_cat', 'code_serv', 'sexe', 'etat_civil',
            'has_children', 'tenure_band', 'career_stage'
        ]
        
        feature_cols = [col for col in feature_cols if col in df_advance.columns]
        
        X = df_advance[feature_cols].copy()
        self.advance_feature_names = X.columns.tolist()  # Store advance model features
        y_class = df_advance['has_advance']
        
        # Encode categorical
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if col in self.label_encoders:
                X[col] = self.label_encoders[col].transform(X[col].astype(str).fillna('Unknown'))
        
        X = X.fillna(X.median())
        
        # CLASSIFICATION
        print("\n--- CLASSIFICATION MODEL ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        
        print(f"Training: {len(X_train):,} | Test: {len(X_test):,}")
        print(f"Advance rate: {y_train.mean()*100:.1f}%")
        
        # Handle class imbalance with balanced weights
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Advance', 'Advance']))
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        self.advance_classifier = clf
        
        # REGRESSION
        df_with_advance = df_advance[df_advance['has_advance'] == 1]
        
        regression_metrics = None
        if len(df_with_advance) >= 50:
            print("\n--- REGRESSION MODEL (Amount Prediction) ---")
            X_reg = df_with_advance[feature_cols].copy()
            y_reg = df_with_advance['advance_total']
            
            for col in X_reg.select_dtypes(include=['object', 'category']).columns:
                if col in self.label_encoders:
                    X_reg[col] = self.label_encoders[col].transform(X_reg[col].astype(str).fillna('Unknown'))
            
            X_reg = X_reg.fillna(X_reg.median())
            
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X_reg, y_reg, test_size=0.2, random_state=42
            )
            
            reg = GradientBoostingRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                min_samples_split=10, random_state=42
            )
            reg.fit(X_train_reg, y_train_reg)
            
            y_pred_reg = reg.predict(X_test_reg)
            
            mae = mean_absolute_error(y_test_reg, y_pred_reg)
            r2 = r2_score(y_test_reg, y_pred_reg)
            mape = np.mean(np.abs((y_test_reg - y_pred_reg) / y_test_reg)) * 100
            
            print(f"MAE: {mae:,.2f} TND | R²: {r2:.4f} | MAPE: {mape:.2f}%")
            
            self.advance_regressor = reg
            regression_metrics = {'mae': mae, 'r2': r2, 'mape': mape}
            
            # Visualizations
            self._plot_advance_diagnostics(
                y_test, y_pred, y_pred_proba, 
                y_test_reg, y_pred_reg, clf
            )
        else:
            print(f"\nInsufficient data for regression ({len(df_with_advance)} samples)")
        
        self.training_metadata['advance'] = {
            'classification': {'accuracy': accuracy},
            'regression': regression_metrics
        }
        
        print("\n" + "="*70)
    
    def _plot_advance_diagnostics(self, y_test_class, y_pred_class, y_pred_proba, 
                                 y_test_reg, y_pred_reg, classifier):
        """Create advance model diagnostic plots"""
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_test_class, y_pred_class)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_title('Confusion Matrix - Advance Classification', fontweight='bold')
        ax1.set_xticklabels(['No Advance', 'Advance'])
        ax1.set_yticklabels(['No Advance', 'Advance'])
        
        # 2. ROC-like: Probability Distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(y_pred_proba[y_test_class == 0], bins=30, alpha=0.7, 
                label='No Advance', edgecolor='black')
        ax2.hist(y_pred_proba[y_test_class == 1], bins=30, alpha=0.7, 
                label='Advance', edgecolor='black')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Probability Distribution by Class', fontweight='bold')
        ax2.legend()
        
        # 3. Feature Importance - Classification
        ax3 = plt.subplot(2, 3, 3)
        if hasattr(classifier, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.advance_feature_names,
                'importance': classifier.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)
            
            ax3.barh(importance_df['feature'], importance_df['importance'])
            ax3.set_xlabel('Importance')
            ax3.set_title('Feature Importance - Classification', fontweight='bold')
        
        # 4. Regression: Actual vs Predicted
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(y_test_reg, y_pred_reg, alpha=0.6, s=50)
        ax4.plot([y_test_reg.min(), y_test_reg.max()], 
                [y_test_reg.min(), y_test_reg.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax4.set_xlabel('Actual Advance Amount (TND)')
        ax4.set_ylabel('Predicted Advance Amount (TND)')
        ax4.set_title('Advance Amount: Actual vs Predicted', fontweight='bold')
        ax4.legend()
        
        # 5. Regression Residuals
        ax5 = plt.subplot(2, 3, 5)
        residuals = y_test_reg - y_pred_reg
        ax5.scatter(y_pred_reg, residuals, alpha=0.6, s=50)
        ax5.axhline(y=0, color='r', linestyle='--', lw=2)
        ax5.set_xlabel('Predicted Amount (TND)')
        ax5.set_ylabel('Residuals (TND)')
        ax5.set_title('Residual Plot - Advance Amount', fontweight='bold')
        
        # 6. Error Distribution
        ax6 = plt.subplot(2, 3, 6)
        error_pct = (np.abs(residuals) / y_test_reg) * 100
        ax6.hist(error_pct, bins=30, edgecolor='black', alpha=0.7)
        ax6.axvline(x=error_pct.median(), color='r', linestyle='--', 
                   lw=2, label=f'Median: {error_pct.median():.1f}%')
        ax6.set_xlabel('Prediction Error (%)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Error Distribution - Advance Amount', fontweight='bold')
        ax6.legend()
        ax6.set_xlim(0, min(error_pct.quantile(0.95), 100))
        
        plt.tight_layout()
        plt.savefig('advance_model_diagnostics.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: advance_model_diagnostics.png")
        plt.close()
    
    def create_business_insights_dashboard(self, df_salary, df_advance):
        """Create comprehensive business insights dashboard"""
        print("\nGenerating business insights dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Salary Growth by Category
        ax1 = plt.subplot(3, 4, 1)
        salary_by_cat = df_salary.groupby('lib_cat')['salary_growth_pct'].mean().sort_values()
        ax1.barh(salary_by_cat.index, salary_by_cat.values)
        ax1.set_xlabel('Avg Salary Growth (%)')
        ax1.set_title('Salary Growth by Job Category', fontweight='bold')
        
        # 2. Salary Distribution by Category
        ax2 = plt.subplot(3, 4, 2)
        df_salary.boxplot(column='salary_total', by='lib_cat', ax=ax2)
        ax2.set_xlabel('Job Category')
        ax2.set_ylabel('Salary (TND)')
        ax2.set_title('Salary Distribution by Category', fontweight='bold')
        plt.sca(ax2)
        plt.xticks(rotation=45, ha='right')
        
        # 3. Advance Rate by Category
        ax3 = plt.subplot(3, 4, 3)
        advance_by_cat = df_advance.groupby('lib_cat')['has_advance'].mean() * 100
        ax3.bar(range(len(advance_by_cat)), advance_by_cat.values)
        ax3.set_xlabel('Job Category')
        ax3.set_ylabel('Advance Rate (%)')
        ax3.set_title('Advance Usage Rate by Category', fontweight='bold')
        ax3.set_xticks(range(len(advance_by_cat)))
        ax3.set_xticklabels(advance_by_cat.index, rotation=45, ha='right')
        
        # 4. Advance Amount by Tenure
        ax4 = plt.subplot(3, 4, 4)
        if 'tenure_band' in df_advance.columns:
            advance_by_tenure = df_advance[df_advance['has_advance']==1].groupby('tenure_band')['advance_total'].mean()
            ax4.bar(range(len(advance_by_tenure)), advance_by_tenure.values)
            ax4.set_xlabel('Tenure Band')
            ax4.set_ylabel('Avg Advance Amount (TND)')
            ax4.set_title('Advance Amount by Tenure', fontweight='bold')
            ax4.set_xticks(range(len(advance_by_tenure)))
            ax4.set_xticklabels(advance_by_tenure.index, rotation=45, ha='right')
        
        # 5. Salary vs Age
        ax5 = plt.subplot(3, 4, 5)
        ax5.scatter(df_salary['age'], df_salary['salary_total'], alpha=0.5, s=30)
        z = np.polyfit(df_salary['age'].dropna(), df_salary['salary_total'][df_salary['age'].notna()], 2)
        p = np.poly1d(z)
        age_range = np.linspace(df_salary['age'].min(), df_salary['age'].max(), 100)
        ax5.plot(age_range, p(age_range), "r-", lw=2, label='Trend')
        ax5.set_xlabel('Age')
        ax5.set_ylabel('Salary (TND)')
        ax5.set_title('Salary vs Age', fontweight='bold')
        ax5.legend()
        
        # 6. Salary vs Tenure
        ax6 = plt.subplot(3, 4, 6)
        ax6.scatter(df_salary['anciennete'], df_salary['salary_total'], alpha=0.5, s=30)
        z = np.polyfit(df_salary['anciennete'].dropna(), 
                      df_salary['salary_total'][df_salary['anciennete'].notna()], 2)
        p = np.poly1d(z)
        tenure_range = np.linspace(df_salary['anciennete'].min(), df_salary['anciennete'].max(), 100)
        ax6.plot(tenure_range, p(tenure_range), "r-", lw=2, label='Trend')
        ax6.set_xlabel('Tenure (years)')
        ax6.set_ylabel('Salary (TND)')
        ax6.set_title('Salary vs Tenure', fontweight='bold')
        ax6.legend()
        
        # 7. Advance Rate by Marital Status
        ax7 = plt.subplot(3, 4, 7)
        advance_by_marital = df_advance.groupby('etat_civil')['has_advance'].mean() * 100
        ax7.bar(range(len(advance_by_marital)), advance_by_marital.values)
        ax7.set_xlabel('Marital Status')
        ax7.set_ylabel('Advance Rate (%)')
        ax7.set_title('Advance Rate by Marital Status', fontweight='bold')
        ax7.set_xticks(range(len(advance_by_marital)))
        ax7.set_xticklabels(advance_by_marital.index, rotation=45, ha='right')
        
        # 8. Advance Rate by Number of Children
        ax8 = plt.subplot(3, 4, 8)
        advance_by_children = df_advance.groupby('nbre_enf')['has_advance'].mean() * 100
        ax8.plot(advance_by_children.index, advance_by_children.values, marker='o', lw=2)
        ax8.set_xlabel('Number of Children')
        ax8.set_ylabel('Advance Rate (%)')
        ax8.set_title('Advance Rate by Family Size', fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # 9. Salary Growth Over Years
        ax9 = plt.subplot(3, 4, 9)
        growth_by_year = df_salary.groupby('annee')['salary_growth_pct'].mean()
        ax9.plot(growth_by_year.index, growth_by_year.values, marker='o', lw=2)
        ax9.set_xlabel('Year')
        ax9.set_ylabel('Avg Salary Growth (%)')
        ax9.set_title('Salary Growth Trend Over Time', fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        # 10. Advance Amount Distribution
        ax10 = plt.subplot(3, 4, 10)
        advance_amounts = df_advance[df_advance['has_advance']==1]['advance_total']
        ax10.hist(advance_amounts, bins=40, edgecolor='black', alpha=0.7)
        ax10.axvline(advance_amounts.median(), color='r', linestyle='--', 
                    lw=2, label=f'Median: {advance_amounts.median():.0f} TND')
        ax10.set_xlabel('Advance Amount (TND)')
        ax10.set_ylabel('Frequency')
        ax10.set_title('Advance Amount Distribution', fontweight='bold')
        ax10.legend()
        
        # 11. Advance Ratio (% of Salary)
        ax11 = plt.subplot(3, 4, 11)
        advance_ratios = df_advance[df_advance['has_advance']==1]['advance_ratio']
        ax11.hist(advance_ratios, bins=40, edgecolor='black', alpha=0.7)
        ax11.axvline(advance_ratios.median(), color='r', linestyle='--', 
                    lw=2, label=f'Median: {advance_ratios.median():.1f}%')
        ax11.set_xlabel('Advance as % of Salary')
        ax11.set_ylabel('Frequency')
        ax11.set_title('Advance Ratio Distribution', fontweight='bold')
        ax11.legend()
        ax11.set_xlim(0, min(advance_ratios.quantile(0.95), 100))
        
        # 12. High Risk Employees
        ax12 = plt.subplot(3, 4, 12)
        risk_summary = df_advance['advance_risk'].value_counts()
        colors_risk = ['#2ecc71', '#f39c12', '#e74c3c']
        ax12.pie(risk_summary.values, labels=risk_summary.index, autopct='%1.1f%%',
                colors=colors_risk, startangle=90)
        ax12.set_title('Advance Risk Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('business_insights_dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: business_insights_dashboard.png")
        plt.close()
    
    def predict_employee_salary(self, employee_data, years=5, confidence_level=0.9):
        """Predict salary with confidence intervals"""
        if not self.salary_model:
            print("Error: Model not trained")
            return None
        
        predictions = []
        current_salary = employee_data['current_salary']
        
        print(f"\n{'='*60}")
        print(f"SALARY FORECAST - {years} YEARS")
        print(f"{'='*60}")
        print(f"Current Salary: {current_salary:,.2f} TND")
        print(f"Category: {employee_data.get('lib_cat', 'N/A')}")
        print(f"Department: {employee_data.get('code_serv', 'N/A')}")
        print(f"Age: {employee_data['age']} | Tenure: {employee_data['anciennete']:.1f} years")
        print("-" * 60)
        print(f"Expected Features: {self.salary_feature_names}")
        
        # Calculate confidence interval (simplified using MAE)
        mae = self.training_metadata.get('salary', {}).get('metrics', {}).get('MAE', 0)
        ci_factor = 1.96 if confidence_level == 0.95 else 1.645  # 95% or 90% CI
        
        for year in range(1, years + 1):
            # Prepare features for prediction
            tenure = employee_data['anciennete'] + year
            tenure_band = pd.cut([tenure], 
                                bins=[-np.inf, 1, 3, 5, 10, np.inf],
                                labels=['<1yr', '1-3yrs', '3-5yrs', '5-10yrs', '10+yrs'])[0]
            career_stage = 'Junior'
            if 3 <= tenure < 10:
                career_stage = 'Mid-level'
            elif tenure >= 10:
                career_stage = 'Senior'
            
            features = {
                'prev_salary': current_salary,
                'age': employee_data['age'] + year,
                'anciennete': tenure,
                'nbre_enf': employee_data.get('nbre_enf', 0),
                'cod_cat': employee_data.get('cod_cat', 'Unknown'),
                'code_serv': employee_data.get('code_serv', 'Unknown'),
                'sexe': employee_data.get('sexe', 'Unknown'),
                'etat_civil': employee_data.get('etat_civil', 'Unknown'),
                'has_children': int(employee_data.get('nbre_enf', 0) > 0),
                'years_employed': int(tenure),
                'dept_avg_salary': employee_data.get('dept_avg_salary', current_salary),
                'salary_vs_dept_avg': (current_salary / employee_data.get('dept_avg_salary', current_salary)) * 100,
                'tenure_band': str(tenure_band),
                'career_stage': str(career_stage)
            }
            
            # Ensure all required features are present
            X = pd.DataFrame(columns=self.salary_feature_names)
            for col in self.salary_feature_names:
                if col in features:
                    X.loc[0, col] = features[col]
                else:
                    X.loc[0, col] = 0  # Fallback
            
            # Encode categorical features
            for col in X.select_dtypes(include=['object', 'category']).columns:
                if col in self.label_encoders:
                    try:
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
                    except ValueError as e:
                        print(f"Error encoding {col}: {e}")
                        X[col] = 0  # Fallback to 0 if encoding fails
            
            X = X.fillna(0)
            
            print(f"Features for Year {year}: {X.columns.tolist()}")
            
            # Verify feature names match
            if set(X.columns) != set(self.salary_feature_names):
                print(f"Warning: Feature mismatch. Expected {self.salary_feature_names}, got {X.columns.tolist()}")
            
            predicted_salary = self.salary_model.predict(X)[0]
            
            # Confidence interval
            lower_bound = predicted_salary - (ci_factor * mae)
            upper_bound = predicted_salary + (ci_factor * mae)
            
            # Confidence interval
            lower_bound = predicted_salary - (ci_factor * mae)
            upper_bound = predicted_salary + (ci_factor * mae)
            
            growth = predicted_salary - current_salary
            growth_pct = (growth / current_salary) * 100
            
            predictions.append({
                'year': year,
                'predicted_salary': predicted_salary,
                'lower_bound': max(0, lower_bound),
                'upper_bound': upper_bound,
                'growth': growth,
                'growth_pct': growth_pct
            })
            
            print(f"Year +{year}: {predicted_salary:>10,.2f} TND "
                  f"[{lower_bound:>10,.2f} - {upper_bound:>10,.2f}] "
                  f"({growth_pct:>+6.2f}%)")
            
            current_salary = predicted_salary
        
        print("="*60)
        
        df_pred = pd.DataFrame(predictions)
        
        # Visualization
        self._plot_salary_forecast(df_pred, employee_data)
        
        return df_pred
    
    def _plot_salary_forecast(self, predictions_df, employee_data):
        """Plot salary forecast with confidence intervals"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        years = predictions_df['year']
        predicted = predictions_df['predicted_salary']
        lower = predictions_df['lower_bound']
        upper = predictions_df['upper_bound']
        
        # Plot 1: Salary Forecast
        ax1.plot(years, predicted, marker='o', lw=2, label='Predicted', color='#3498db')
        ax1.fill_between(years, lower, upper, alpha=0.3, color='#3498db', 
                         label='90% Confidence Interval')
        ax1.set_xlabel('Years Ahead')
        ax1.set_ylabel('Salary (TND)')
        ax1.set_title(f'Salary Forecast - {employee_data.get("lib_cat", "Employee")}', 
                     fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Growth Rate
        growth_pct = predictions_df['growth_pct']
        ax2.bar(years, growth_pct, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Years Ahead')
        ax2.set_ylabel('Growth Rate (%)')
        ax2.set_title('Year-over-Year Growth Rate', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', lw=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('salary_forecast.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: salary_forecast.png")
        plt.close()
    
    def predict_advance_risk(self, employee_data):
        """Enhanced advance risk prediction"""
        if not self.advance_classifier:
            print("Error: Model not trained")
            return None
        
        tenure = employee_data['anciennete']
        tenure_band = pd.cut([tenure], 
                            bins=[-np.inf, 1, 3, 5, 10, np.inf],
                            labels=['<1yr', '1-3yrs', '3-5yrs', '5-10yrs', '10+yrs'])[0]
        career_stage = 'Junior'
        if 3 <= tenure < 10:
            career_stage = 'Mid-level'
        elif tenure >= 10:
            career_stage = 'Senior'
        
        features = {
            'salary': employee_data['current_salary'],
            'age': employee_data['age'],
            'anciennete': tenure,
            'nbre_enf': employee_data.get('nbre_enf', 0),
            'cod_cat': employee_data.get('cod_cat', 'Unknown'),
            'code_serv': employee_data.get('code_serv', 'Unknown'),
            'sexe': employee_data.get('sexe', 'Unknown'),
            'etat_civil': employee_data.get('etat_civil', 'Unknown'),
            'has_children': int(employee_data.get('nbre_enf', 0) > 0),
            'tenure_band': str(tenure_band),
            'career_stage': career_stage
        }
        
        # Ensure all required features are present
        X = pd.DataFrame(columns=self.advance_feature_names)
        for col in self.advance_feature_names:
            if col in features:
                X.loc[0, col] = features[col]
            else:
                X.loc[0, col] = 0
        
        for col in X.select_dtypes(include=['object']).columns:
            if col in self.label_encoders:
                try:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
                except ValueError as e:
                    print(f"Error encoding {col}: {e}")
                    X[col] = 0
        
        X = X.fillna(0)
        
        print(f"Advance Prediction Features: {X.columns.tolist()}")
        
        # Verify feature names match
        if set(X.columns) != set(self.advance_feature_names):
            print(f"Warning: Feature mismatch for advance. Expected {self.advance_feature_names}, got {X.columns.tolist()}")
        
        risk_prob = self.advance_classifier.predict_proba(X)[0][1]
        risk_level = 'High' if risk_prob > 0.7 else 'Medium' if risk_prob > 0.4 else 'Low'
        
        result = {
            'risk_probability': risk_prob,
            'risk_level': risk_level,
            'recommendation': ''
        }
        
        print(f"\n{'='*60}")
        print("ADVANCE RISK ASSESSMENT")
        print(f"{'='*60}")
        print(f"Employee: {employee_data.get('lib_cat', 'N/A')}")
        print(f"Salary: {employee_data['current_salary']:,.2f} TND")
        print(f"\nRisk Probability: {risk_prob*100:.1f}%")
        print(f"Risk Level: {risk_level}")
        
        if self.advance_regressor and risk_prob > 0.3:
            predicted_amount = self.advance_regressor.predict(X)[0]
            result['predicted_amount'] = predicted_amount
            result['predicted_ratio'] = (predicted_amount / employee_data['current_salary']) * 100
            print(f"Predicted Amount: {predicted_amount:,.2f} TND ({result['predicted_ratio']:.1f}% of salary)")
        
        # Recommendations
        if risk_level == 'High':
            result['recommendation'] = "Monitor closely. Consider financial counseling or budget planning support."
        elif risk_level == 'Medium':
            result['recommendation'] = "Moderate risk. Standard monitoring procedures apply."
        else:
            result['recommendation'] = "Low risk. No special action required."
        
        print(f"\nRecommendation: {result['recommendation']}")
        print("="*60)
        
        return result
    
    def save_models(self, filename='enhanced_payroll_models.pkl'):
        """Save all models and metadata"""
        model_data = {
            'salary_model': self.salary_model,
            'advance_classifier': self.advance_classifier,
            'advance_regressor': self.advance_regressor,
            'label_encoders': self.label_encoders,
            'salary_feature_names': self.salary_feature_names,
            'advance_feature_names': self.advance_feature_names,
            'training_metadata': self.training_metadata,
            'department_stats': self.department_stats,
            'category_stats': self.category_stats,
            'scaler': self.scaler
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Models saved: {filename}")
        print(f"  Size: {os.path.getsize(filename) / 1024:.1f} KB")
    
    def generate_summary_report(self, df_salary, df_advance):
        """Generate text summary report"""
        print("\n" + "="*70)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*70)
        
        print("\n1. SALARY PREDICTION MODEL")
        print("-" * 70)
        salary_meta = self.training_metadata.get('salary', {}).get('metrics', {})
        print(f"   R² Score: {salary_meta.get('R2', 0):.4f}")
        print(f"   MAE: {salary_meta.get('MAE', 0):,.2f} TND")
        print(f"   MAPE: {salary_meta.get('MAPE', 0):.2f}%")
        print(f"   Training Samples: {self.training_metadata.get('salary', {}).get('n_samples', 0):,}")
        
        print("\n2. ADVANCE PREDICTION MODELS")
        print("-" * 70)
        print(f"   Classification Accuracy: Available")
        print(f"   Regression MAE: Available")
        print(f"   Total Employees Analyzed: {len(df_advance):,}")
        print(f"   Advance Rate: {df_advance['has_advance'].mean()*100:.1f}%")
        
        print("\n3. KEY INSIGHTS")
        print("-" * 70)
        print(f"   Average Salary: {df_salary['salary_total'].mean():,.2f} TND")
        print(f"   Average Growth Rate: {df_salary['salary_growth_pct'].mean():.2f}%")
        print(f"   Average Advance Amount: {df_advance[df_advance['has_advance']==1]['advance_total'].mean():,.2f} TND")
        print(f"   High Risk Employees: {(df_advance['advance_risk']=='High').sum():,}")
        
        print("\n" + "="*70)
    
    def train(self, year=None):
        """Train the models and return metrics"""
        df = self.load_data(year)
        df = self.categorize_payments(df)
        df = self.engineer_features(df)
        
        self.calculate_department_benchmarks(df)
        
        self.df_salary = self.prepare_salary_data(df)
        self.df_advance = self.prepare_advance_data(df)
        
        self.fit_label_encoders(pd.concat([self.df_salary, self.df_advance]))
        
        self.train_salary_model(self.df_salary)
        self.train_advance_models(self.df_advance)
        
        self.create_business_insights_dashboard(self.df_salary, self.df_advance)
        self.generate_summary_report(self.df_salary, self.df_advance)
        self.save_models()
        
        salary_metrics = self.training_metadata['salary']['metrics']
        
        advance_metrics = self.training_metadata.get('advance', {'classification': {'accuracy': 0}, 'regression': None})
        
        feature_importance = []
        if hasattr(self.salary_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.salary_feature_names,
                'importance': self.salary_model.feature_importances_
            }).sort_values('importance', ascending=False).to_dict('records')
            feature_importance = importance_df
        
        return {
            'success': True,
            'salary_metrics': {
                'r2': salary_metrics['R2'],
                'mae': salary_metrics['MAE']
            },
            'advance_metrics': advance_metrics['classification'],
            'training_samples': len(self.df_salary),
            'feature_importance': feature_importance
        }
    
    def get_salary_progression(self, year=None):
        """Get salary progression data"""
        if self.df_salary is None:
            raise Exception("Model not trained")
        
        data = {
            'avg_growth_pct': self.df_salary['salary_growth_pct'].mean(),
            'median_growth_pct': self.df_salary['salary_growth_pct'].median(),
            'avg_growth_amount': self.df_salary['salary_growth'].mean(),
            'total_employees': self.df_salary['mat_pers'].nunique(),
            'by_category': {},
            'by_tenure': {}
        }
        
        for cat, group in self.df_salary.groupby('cod_cat'):
            data['by_category'][cat] = {
                'avg_growth': group['salary_growth_pct'].mean(),
                'count': len(group)
            }
        
        for tenure, group in self.df_salary.groupby('tenure_band'):
            data['by_tenure'][tenure] = {
                'avg_growth': group['salary_growth_pct'].mean(),
                'count': len(group)
            }
        
        return data
    
    def get_advance_analysis(self, year=None):
        """Get advance analysis data"""
        if self.df_advance is None:
            raise Exception("Model not trained")
        
        self.df_advance['salary_bracket'] = pd.qcut(self.df_advance['salary'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        
        data = {
            'advance_rate': self.df_advance['has_advance'].mean() * 100,
            'employees_with_advances': self.df_advance['has_advance'].sum(),
            'total_employees': len(self.df_advance),
            'avg_advance_amount': self.df_advance[self.df_advance['has_advance'] == 1]['advance_total'].mean(),
            'total_advances': self.df_advance['advance_total'].sum(),
            'risk_by_salary_bracket': (self.df_advance.groupby('salary_bracket')['has_advance'].mean() * 100).to_dict(),
            'risk_by_children': (self.df_advance.groupby('nbre_enf')['has_advance'].mean() * 100).to_dict()
        }
        
        return data
    
    def get_employees(self, year=None):
        """Get list of employees"""
        query = """
        SELECT DISTINCT mat_pers, code_serv AS service
        FROM DIM_EMPLOYEe_nouveau
        """
        df = pd.read_sql(query, self.engine)
        return df.to_dict('records')
    
    def predict_employee(self, mat_pers, year=None):
        """Predict for specific employee"""
        # Query employee data
        emp_query = f"""
        SELECT sexe, age, anciennete, etat_civil, nbre_enf, cod_cat,
               code_serv, lib_cat
        FROM DIM_EMPLOYEe_nouveau
        WHERE mat_pers = '{mat_pers}'
        """
        emp_df = pd.read_sql(emp_query, self.engine)
        if emp_df.empty:
            raise ValueError("Employee not found")
        
        emp = emp_df.iloc[0]
        
        # Get current salary (SNET)
        salary_query = f"""
        SELECT montant 
        FROM FAIT_remuneration fr
        JOIN DIM_TEMPS_nouveau dt ON fr.id_temps = dt.id_temps
        WHERE mat_pers = '{mat_pers}' AND LOWER(source) LIKE '%snet%'
        ORDER BY dt.annee DESC, dt.mois DESC
        LIMIT 1
        """
        salary_df = pd.read_sql(salary_query, self.engine)
        current_salary = float(salary_df['montant'].iloc[0]) if not salary_df.empty else 0
        
        employee_data = {
            'current_salary': current_salary,
            'age': float(emp['age']),
            'anciennete': float(emp['anciennete']),
            'nbre_enf': int(emp['nbre_enf']),
            'cod_cat': emp['cod_cat'],
            'code_serv': emp['code_serv'],
            'sexe': emp['sexe'],
            'etat_civil': emp['etat_civil'],
            'lib_cat': emp['lib_cat'],
            'dept_avg_salary': self.department_stats.get(emp['code_serv'], {}).get('avg_salary', current_salary)
        }
        
        salary_pred = self.predict_employee_salary(employee_data, years=3)
        
        advance_risk = self.predict_advance_risk(employee_data)
        
        return {
            'employee_info': {
                'mat_pers': mat_pers,
                'current_salary': employee_data['current_salary'],
                'age': employee_data['age'],
                'anciennete': employee_data['anciennete']
            },
            'salary_predictions': salary_pred.to_dict('records'),
            'advance_risk': advance_risk
        }

model = EnhancedPayrollPredictiveModel()

@prediction_bp.route('/train', methods=['GET'])
def train():
    year = request.args.get('year', default=None, type=int)
    try:
        metrics = model.train(year)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@prediction_bp.route('/salary-progression', methods=['GET'])
def salary_progression():
    year = request.args.get('year', default=None, type=int)
    try:
        data = model.get_salary_progression(year)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@prediction_bp.route('/advance-analysis', methods=['GET'])
def advance_analysis():
    year = request.args.get('year', default=None, type=int)
    try:
        data = model.get_advance_analysis(year)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@prediction_bp.route('/employees', methods=['GET'])
def employees():
    year = request.args.get('year', default=None, type=int)
    try:
        emps = model.get_employees(year)
        return jsonify(emps)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@prediction_bp.route('/employee/<mat_pers>', methods=['GET'])
def employee_prediction(mat_pers):
    year = request.args.get('year', default=None, type=int)
    try:
        data = model.predict_employee(mat_pers, year)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
