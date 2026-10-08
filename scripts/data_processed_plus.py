import logging
import pandas as pd
import numpy as np
from pathlib import Path


from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------

# ------------------------
config = {
    'input_path': 'Baza customer Telecom v2.csv',
    'output_path': 'processed_customer_data_plus_improved.csv',
    'target_column': 'CHURN',
    'target_map': {'No': 0, 'Yes': 1},
    'invalid_value_thresholds': {'Total_Subscribers': 1},
    'knn_imputer_params': {'n_neighbors': 5, 'weights': 'uniform'},
    'log_transform': False,  
    'outlier_params': {'lower_quantile': 0.01, 'upper_quantile': 0.99},
    'derived_numeric_features': [
        'Revenue_Per_Subscriber',
        'Mobile_to_Fixed_Ratio',
        'Suspended_Ratio',
        'Inactive_Ratio',
        'Segment_Churn_Rate',
        'ZIP_Churn_Rate',
        'ZIP_ARPU_Avg'
    ],
    'smote_params': {'random_state': 42},
    
    'arpu_bins': 4,  
    'segment_column': 'CRM_PID_Value_Segment',  
    'zip_column': 'Billing_ZIP',  
    'arpu_column': 'ARPU',  
    'suspended_column': 'Suspended_subscribers',  
    'inactive_column': 'Not_Active_subscribers',  
    'min_segment_size': 10,  
    'smoothing_factor': 0.1  
}

# ------------------------

# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------

# ------------------------
class InitialCleaner(BaseEstimator, TransformerMixin):
    """Remove duplicates, constant or empty columns, and rows failing the configured validity threshold."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        before = df.shape[0]
        df = df.drop_duplicates()
        logger.info(f"Dropped {before - df.shape[0]} duplicate rows")

        nunique = df.nunique()
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            df = df.drop(columns=const_cols)
            logger.info(f"Dropped constant columns: {const_cols}")

        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            df = df.drop(columns=null_cols)
            logger.info(f"Dropped fully-null columns: {null_cols}")

        for col, min_val in config['invalid_value_thresholds'].items():
            if col in df.columns:
                before = df.shape[0]
                df = df[df[col] >= min_val]
                logger.info(f"Dropped {before - df.shape[0]} rows where {col} < {min_val}")

        return df

class EnhancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Construct segment-level target encoding, suspended/inactive ratios, regional aggregates and ARPU bins."""
    def __init__(self, target_column='CHURN', target_map=None):
        self.target_column = target_column
        self.target_map = target_map or {'No': 0, 'Yes': 1}
        self.segment_stats_ = {}
        self.zip_stats_ = {}
        self.arpu_bins_ = None
        self.global_churn_rate_ = None
    
    def fit(self, X, y=None):
        """Estimate feature-engineering statistics from the supplied fitting data."""
        df = X.copy()
        
        
        if self.target_column in df.columns:
            
            target_numeric = df[self.target_column].map(self.target_map)
            self.global_churn_rate_ = target_numeric.mean()
            
            
            seg_col = config.get('segment_column')
            if seg_col in df.columns:
                segment_stats = df.groupby(seg_col).agg({
                    self.target_column: lambda x: x.map(self.target_map).mean(),
                    seg_col: 'count'
                }).rename(columns={seg_col: 'count'})
                
                
                smoothing = config.get('smoothing_factor', 0.1)
                min_size = config.get('min_segment_size', 10)
                
                segment_stats['smoothed_rate'] = (
                    segment_stats[self.target_column] * segment_stats['count'] + 
                    self.global_churn_rate_ * smoothing
                ) / (segment_stats['count'] + smoothing)
                
                
                segment_stats.loc[segment_stats['count'] < min_size, 'smoothed_rate'] = self.global_churn_rate_
                
                self.segment_stats_ = segment_stats['smoothed_rate'].to_dict()
                logger.info(f"Computed segment churn rates for {len(self.segment_stats_)} segments")
            
            
            zip_col = config.get('zip_column')
            arpu_col = config.get('arpu_column')
            if zip_col in df.columns:
                zip_agg = {self.target_column: lambda x: x.map(self.target_map).mean()}
                if arpu_col in df.columns:
                    zip_agg[arpu_col] = 'mean'
                
                zip_stats = df.groupby(zip_col).agg(zip_agg)
                zip_stats.columns = ['churn_rate', 'arpu_avg'] if arpu_col in df.columns else ['churn_rate']
                
                self.zip_stats_ = zip_stats.to_dict('index')
                logger.info(f"Computed ZIP-level stats for {len(self.zip_stats_)} regions")
        
        
        arpu_col = config.get('arpu_column')
        if arpu_col in df.columns:
            
            try:
                self.arpu_bins_ = pd.qcut(df[arpu_col].dropna(), 
                                        q=config.get('arpu_bins', 4), 
                                        duplicates='drop',
                                        precision=2)
                
                self.arpu_bin_edges_ = self.arpu_bins_.cat.categories
                logger.info(f"Created ARPU bins: {list(self.arpu_bin_edges_)}")
            except Exception as e:
                logger.warning(f"Failed to create ARPU bins: {e}")
                self.arpu_bins_ = None
        
        return self
    
    def transform(self, X):
        """Apply the fitted feature transformations."""
        df = X.copy()
        
        
        if {'Total_Revenue', 'Total_Subscribers'}.issubset(df.columns):
            df['Revenue_Per_Subscriber'] = df['Total_Revenue'] / (df['Total_Subscribers'] + 1e-6)
        
        if {'Mobile_Revenue', 'Fixed_Revenue'}.issubset(df.columns):
            df['Mobile_to_Fixed_Ratio'] = df['Mobile_Revenue'] / (df['Fixed_Revenue'] + 1e-6)
        
        
        seg_col = config.get('segment_column')
        if seg_col in df.columns and self.segment_stats_:
            df['Segment_Churn_Rate'] = df[seg_col].map(self.segment_stats_).fillna(self.global_churn_rate_)
            logger.info("Added Segment_Churn_Rate feature")
        
        
        suspended_col = config.get('suspended_column')
        inactive_col = config.get('inactive_column')
        
        if suspended_col in df.columns and 'Total_Subscribers' in df.columns:
            df['Suspended_Ratio'] = df[suspended_col] / (df['Total_Subscribers'] + 1e-6)
            logger.info("Added Suspended_Ratio feature")
        
        if inactive_col in df.columns and 'Total_Subscribers' in df.columns:
            df['Inactive_Ratio'] = df[inactive_col] / (df['Total_Subscribers'] + 1e-6)
            logger.info("Added Inactive_Ratio feature")
        
        
        zip_col = config.get('zip_column')
        if zip_col in df.columns and self.zip_stats_:
            df['ZIP_Churn_Rate'] = df[zip_col].map(
                lambda x: self.zip_stats_.get(x, {}).get('churn_rate', self.global_churn_rate_)
            )
            
            if 'arpu_avg' in list(self.zip_stats_.values())[0]:
                df['ZIP_ARPU_Avg'] = df[zip_col].map(
                    lambda x: self.zip_stats_.get(x, {}).get('arpu_avg', df[config.get('arpu_column', 'ARPU')].mean())
                )
                logger.info("Added ZIP_Churn_Rate and ZIP_ARPU_Avg features")
            else:
                logger.info("Added ZIP_Churn_Rate feature")
        
        
        arpu_col = config.get('arpu_column')
        if arpu_col in df.columns and self.arpu_bins_ is not None:
            try:
                
                df['ARPU_Bin'] = pd.cut(df[arpu_col], 
                                      bins=self.arpu_bin_edges_, 
                                      include_lowest=True, 
                                      labels=range(len(self.arpu_bin_edges_)-1))
                
                df['ARPU_Bin'] = pd.to_numeric(df['ARPU_Bin'], errors='coerce').fillna(0).astype(int)
                logger.info("Added ARPU_Bin feature")
            except Exception as e:
                logger.warning(f"Failed to apply ARPU binning: {e}")
        
        
        if 'Revenue_Per_Subscriber' in df.columns and 'Suspended_Ratio' in df.columns:
            df['RPS_Suspended_Interaction'] = df['Revenue_Per_Subscriber'] * df['Suspended_Ratio']
        
        if 'Segment_Churn_Rate' in df.columns and 'ZIP_Churn_Rate' in df.columns:
            df['Segment_ZIP_Churn_Avg'] = (df['Segment_Churn_Rate'] + df['ZIP_Churn_Rate']) / 2
        
        return df

class OutlierCapper(BaseEstimator, TransformerMixin):
    """Clip feature values using configured quantile bounds."""
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.bounds_ = []
        for i in range(arr.shape[1]):
            col = arr[:, i]
            valid_col = col[~np.isnan(col)]  
            if len(valid_col) > 0:
                low = np.quantile(valid_col, self.lower_quantile)
                high = np.quantile(valid_col, self.upper_quantile)
            else:
                low, high = 0, 1  
            self.bounds_.append((low, high))
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        out = arr.copy()
        for i, (low, high) in enumerate(self.bounds_):
            out[:, i] = np.clip(out[:, i], low, high)
        return out

class DynamicPreprocessor(BaseEstimator, TransformerMixin):
    """Determine column types at runtime and construct numeric and categorical preprocessing."""
    def __init__(self, knn_imputer_params=None, log_transform=True, outlier_params=None):
        self.knn_imputer_params = knn_imputer_params or {}
        self.log_transform = log_transform
        self.outlier_params = outlier_params or {}
    
    def fit(self, X, y=None):
        
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        
        if config['target_column'] in df.columns:
            df = df.drop(columns=[config['target_column']])
        
        self.num_cols_ = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols_ = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        
        high_cardinality_cols = []
        for col in self.cat_cols_:
            if df[col].nunique() > 50:  
                high_cardinality_cols.append(col)
                logger.warning(f"High cardinality column {col} has {df[col].nunique()} unique values")
        
        
        for col in high_cardinality_cols:
            if col in [config.get('zip_column'), config.get('segment_column')]:
                
                self.cat_cols_.remove(col)
                logger.info(f"Removed high cardinality column {col} (handled by target encoding)")
            else:
                self.cat_cols_.remove(col)
                logger.info(f"Removed high cardinality column {col} to prevent feature explosion")
        
        logger.info(f"Numeric columns: {len(self.num_cols_)}")
        logger.info(f"Categorical columns: {len(self.cat_cols_)}")
        
        
        num_pipeline = Pipeline([
            ('imputer', KNNImputer(**self.knn_imputer_params)),
            ('log', FunctionTransformer(func=np.log1p, validate=False)
                 if self.log_transform
                 else FunctionTransformer(func=lambda x: x, validate=False)),
            ('outlier', OutlierCapper(**self.outlier_params)),
            ('scaler', StandardScaler())
        ])
        
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        
        transformers = []
        if self.num_cols_:
            transformers.append(('num', num_pipeline, self.num_cols_))
        if self.cat_cols_:
            transformers.append(('cat', cat_pipeline, self.cat_cols_))
        
        if not transformers:
            raise ValueError("No valid columns found for preprocessing")
        
        self.preprocessor_ = ColumnTransformer(transformers, remainder='drop')
        
        
        self.preprocessor_.fit(df)
        
        return self
    
    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        
        if config['target_column'] in df.columns:
            df = df.drop(columns=[config['target_column']])
        
        return self.preprocessor_.transform(df)
    
    def get_feature_names_out(self, input_features=None):
        """Return output feature names."""
        feature_names = []
        
        
        if self.num_cols_:
            feature_names.extend(self.num_cols_)
        
        
        if self.cat_cols_:
            try:
                ohe = self.preprocessor_.named_transformers_['cat'].named_steps['onehot']
                cat_features = ohe.get_feature_names_out(self.cat_cols_).tolist()
                feature_names.extend(cat_features)
            except Exception:
                
                feature_names.extend([f'cat_{i}' for i in range(len(self.cat_cols_))])
        
        return feature_names

# ------------------------

# ------------------------
def build_preprocessing_pipeline(df: pd.DataFrame):
    
    y = df[config['target_column']].map(config['target_map'])
    X0 = df.copy()  

    
    full_pipeline = Pipeline([
        ('clean', InitialCleaner()),
        ('enhanced_fe', EnhancedFeatureEngineer(
            target_column=config['target_column'],
            target_map=config['target_map']
        )),
        ('prep', DynamicPreprocessor(
            knn_imputer_params=config['knn_imputer_params'],
            log_transform=config['log_transform'],
            outlier_params=config['outlier_params']
        )),
        ('smote', SMOTE(**config['smote_params']))
    ])

    return full_pipeline, X0, y

# ------------------------

# ------------------------
def main():
    
    df = pd.read_csv(Path(config['input_path']))
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} cols")
    
    
    logger.info(f"Target distribution: {df[config['target_column']].value_counts().to_dict()}")
    
    
    key_columns = [
        config.get('segment_column'),
        config.get('zip_column'),
        config.get('arpu_column'),
        config.get('suspended_column'),
        config.get('inactive_column')
    ]
    
    existing_columns = [col for col in key_columns if col and col in df.columns]
    missing_columns = [col for col in key_columns if col and col not in df.columns]
    
    logger.info(f"Existing key columns: {existing_columns}")
    if missing_columns:
        logger.warning(f"Missing key columns: {missing_columns}")

    pipeline, X0, y = build_preprocessing_pipeline(df)
    X_res, y_res = pipeline.fit_resample(X0, y)
    logger.info(f"After resampling: {X_res.shape[0]} samples, {X_res.shape[1]} features")

    
    try:
        feature_names = pipeline.named_steps['prep'].get_feature_names_out()
    except Exception as e:
        logger.warning(f"Could not get feature names: {e}")
        feature_names = [f'feature_{i}' for i in range(X_res.shape[1])]

    processed_df = pd.DataFrame(X_res, columns=feature_names)
    processed_df[config['target_column']] = y_res
    processed_df.to_csv(config['output_path'], index=False)
    logger.info(f"Processed data saved to {config['output_path']}")
    logger.info(f"Final shape: {processed_df.shape}")
    logger.info(f"Target distribution: {processed_df[config['target_column']].value_counts().to_dict()}")
    
    
    new_features = [
        'Segment_Churn_Rate', 'Suspended_Ratio', 'Inactive_Ratio',
        'ZIP_Churn_Rate', 'ZIP_ARPU_Avg', 'ARPU_Bin'
    ]
    
    logger.info("=== New Features Summary ===")
    for feat in new_features:
        if feat in processed_df.columns:
            logger.info(f"{feat}: mean={processed_df[feat].mean():.4f}, std={processed_df[feat].std():.4f}")

if __name__ == '__main__':
    main()