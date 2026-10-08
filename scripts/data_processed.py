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
    'output_path': 'processed_customer_data_improved.csv',
    'target_column': 'CHURN',
    'target_map': {'No': 0, 'Yes': 1},
    'invalid_value_thresholds': {'Total_Subscribers': 1},
    'knn_imputer_params': {'n_neighbors': 5, 'weights': 'uniform'},
    'log_transform': True,
    'outlier_params': {'lower_quantile': 0.01, 'upper_quantile': 0.99},
    'derived_numeric_features': [
        'Revenue_Per_Subscriber',
        'Mobile_to_Fixed_Ratio'
    ],
    'smote_params': {'random_state': 42}
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

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Construct revenue-per-user and mobile-to-fixed-revenue ratios."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        df = X.copy()
        if {'Total_Revenue', 'Total_Subscribers'}.issubset(df.columns):
            df['Revenue_Per_Subscriber'] = df['Total_Revenue'] / (df['Total_Subscribers'] + 1e-6)
        if {'Mobile_Revenue', 'Fixed_Revenue'}.issubset(df.columns):
            df['Mobile_to_Fixed_Ratio'] = df['Mobile_Revenue'] / (df['Fixed_Revenue'] + 1e-6)
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
            low = np.quantile(col, self.lower_quantile)
            high = np.quantile(col, self.upper_quantile)
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
        self.num_cols_ = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols_ = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        
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
        
        self.preprocessor_ = ColumnTransformer(transformers, remainder='drop')
        
        
        self.preprocessor_.fit(X)
        
        return self
    
    def transform(self, X):
        return self.preprocessor_.transform(X)
    
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
    X0 = df.drop(columns=[config['target_column']])

    
    full_pipeline = Pipeline([
        ('clean', InitialCleaner()),
        ('fe', FeatureEngineer()),
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

if __name__ == '__main__':
    main()