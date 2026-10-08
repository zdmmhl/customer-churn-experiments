import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------

# ------------------------
config = {
    'input_path': 'Baza customer Telecom v2.csv',
    'output_path': 'processed_customer_data_plus_nosmote.csv',
    'target_column': 'CHURN',
    'target_map': {'No': 0, 'Yes': 1},
    'invalid_value_thresholds': {'Total_Subscribers': 1},
    'knn_imputer_params': {'n_neighbors': 5, 'weights': 'uniform'},
    'log_transform': False,
    'outlier_params': {'lower_quantile': 0.01, 'upper_quantile': 0.99},
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
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        before = df.shape[0]
        df = df.drop_duplicates()
        logger.info(f'Dropped {before - df.shape[0]} duplicate rows')

        nunique = df.nunique()
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            df = df.drop(columns=const_cols)
            logger.info(f'Dropped constant columns: {const_cols}')

        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            df = df.drop(columns=null_cols)
            logger.info(f'Dropped fully-null columns: {null_cols}')

        for col, min_val in config['invalid_value_thresholds'].items():
            if col in df.columns:
                before = df.shape[0]
                df = df[df[col] >= min_val]
                logger.info(f'Dropped {before - df.shape[0]} rows where {col} < {min_val}')
        return df

class EnhancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='CHURN', target_map=None):
        self.target_column = target_column
        self.target_map = target_map or {'No': 0, 'Yes': 1}
        self.segment_stats_ = {}
        self.zip_stats_ = {}
        self.global_churn_rate_ = None
        self.arpu_bins_ = None
        self.arpu_bin_edges_ = None

    def fit(self, X, y=None):
        df = X.copy()
        if self.target_column in df.columns:
            target_numeric = df[self.target_column].map(self.target_map)
            self.global_churn_rate_ = target_numeric.mean()

            
            seg_col = config.get('segment_column')
            if seg_col in df.columns:
                stats = df.groupby(seg_col)[self.target_column].apply(lambda x: x.map(self.target_map).mean())
                counts = df.groupby(seg_col).size()
                smoothing = config.get('smoothing_factor')
                min_size = config.get('min_segment_size')
                smoothed = ((stats * counts) + (self.global_churn_rate_ * smoothing)) / (counts + smoothing)
                smoothed[counts < min_size] = self.global_churn_rate_
                self.segment_stats_ = smoothed.to_dict()

            
            zip_col = config.get('zip_column')
            arpu_col = config.get('arpu_column')
            if zip_col in df.columns:
                agg = {self.target_column: lambda x: x.map(self.target_map).mean()}
                if arpu_col in df.columns:
                    agg[arpu_col] = 'mean'
                zip_stats = df.groupby(zip_col).agg(agg)
                cols = ['churn_rate', 'arpu_avg'] if arpu_col in df.columns else ['churn_rate']
                zip_stats.columns = cols
                self.zip_stats_ = zip_stats.to_dict('index')

            
            if arpu_col in df.columns:
                try:
                    bins = pd.qcut(df[arpu_col].dropna(), q=config.get('arpu_bins'), duplicates='drop')
                    self.arpu_bin_edges_ = bins.cat.categories
                except Exception as e:
                    logger.warning(f'ARPU binning failed: {e}')
        return self

    def transform(self, X):
        df = X.copy()
        
        if {'Total_Revenue', 'Total_Subscribers'}.issubset(df.columns):
            df['Revenue_Per_Subscriber'] = df['Total_Revenue'] / (df['Total_Subscribers'] + 1e-6)
        
        sus = config.get('suspended_column')
        ina = config.get('inactive_column')
        if sus in df.columns and 'Total_Subscribers' in df.columns:
            df['Suspended_Ratio'] = df[sus] / (df['Total_Subscribers'] + 1e-6)
        if ina in df.columns and 'Total_Subscribers' in df.columns:
            df['Inactive_Ratio'] = df[ina] / (df['Total_Subscribers'] + 1e-6)

        
        seg_col = config.get('segment_column')
        if seg_col in df.columns:
            df['Segment_Churn_Rate'] = df[seg_col].map(self.segment_stats_).fillna(self.global_churn_rate_)

        
        zip_col = config.get('zip_column')
        if zip_col in df.columns:
            df['ZIP_Churn_Rate'] = df[zip_col].map(lambda x: self.zip_stats_.get(x, {}).get('churn_rate', self.global_churn_rate_))
            if 'arpu_avg' in next(iter(self.zip_stats_.values()), {}):
                df['ZIP_ARPU_Avg'] = df[zip_col].map(lambda x: self.zip_stats_.get(x, {}).get('arpu_avg', df[config.get('arpu_column')].mean()))

        
        arpu_col = config.get('arpu_column')
        if arpu_col in df.columns and self.arpu_bin_edges_ is not None:
            arpu_bins = pd.cut(
            df[arpu_col],
            bins=self.arpu_bin_edges_,
            include_lowest=True
        )            
        codes = arpu_bins.codes                 
        codes[codes < 0] = 0                    
        df['ARPU_Bin'] = codes.astype(int)      

        
        if 'Revenue_Per_Subscriber' in df.columns and 'Suspended_Ratio' in df.columns:
            df['RPS_Suspended_Interaction'] = df['Revenue_Per_Subscriber'] * df['Suspended_Ratio']
        if 'Segment_Churn_Rate' in df.columns and 'ZIP_Churn_Rate' in df.columns:
            df['Segment_ZIP_Churn_Avg'] = (df['Segment_Churn_Rate'] + df['ZIP_Churn_Rate']) / 2
        return df

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.bounds_ = []
        for col in arr.T:
            valid = col[~np.isnan(col)]
            low = np.quantile(valid, self.lower_quantile) if len(valid) else 0
            high = np.quantile(valid, self.upper_quantile) if len(valid) else 1
            self.bounds_.append((low, high))
        return self
    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        for i, (low, high) in enumerate(self.bounds_):
            arr[:, i] = np.clip(arr[:, i], low, high)
        return arr

class DynamicPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, knn_imputer_params=None, log_transform=True, outlier_params=None):
        self.knn_imputer_params = knn_imputer_params or {}
        self.log_transform = log_transform
        self.outlier_params = outlier_params or {}
    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        if config['target_column'] in df.columns:
            df = df.drop(columns=[config['target_column']])
        self.num_cols_ = df.select_dtypes(include=['int64','float64']).columns.tolist()
        self.cat_cols_ = df.select_dtypes(include=['object','category']).columns.tolist()
        
        num_pipe = Pipeline([
            ('imputer', KNNImputer(**self.knn_imputer_params)),
            ('log', FunctionTransformer(np.log1p, validate=False) if self.log_transform else FunctionTransformer(lambda x: x, validate=False)),
            ('outlier', OutlierCapper(**self.outlier_params)),
            ('scaler', StandardScaler())
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', fill_value='Unknown')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers = []
        if self.num_cols_:
            transformers.append(('num', num_pipe, self.num_cols_))
        if self.cat_cols_:
            transformers.append(('cat', cat_pipe, self.cat_cols_))
        self.preprocessor_ = ColumnTransformer(transformers, remainder='drop')
        self.preprocessor_.fit(df)
        return self
    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        if config['target_column'] in df.columns:
            df = df.drop(columns=[config['target_column']])
        return self.preprocessor_.transform(df)
    def get_feature_names_out(self, input_features=None):
        names = []
        if self.num_cols_:
            names.extend(self.num_cols_)
        if self.cat_cols_:
            ohe = self.preprocessor_.named_transformers_['cat'].named_steps['onehot']
            names.extend(ohe.get_feature_names_out(self.cat_cols_))
        return names


def build_preprocessing_pipeline(df):
    y = df[config['target_column']].map(config['target_map'])
    X0 = df.copy()
    pipeline = Pipeline([
        ('clean', InitialCleaner()),
        ('enhanced_fe', EnhancedFeatureEngineer(target_column=config['target_column'], target_map=config['target_map'])),
        ('prep', DynamicPreprocessor(knn_imputer_params=config['knn_imputer_params'], log_transform=config['log_transform'], outlier_params=config['outlier_params']))
    ])
    return pipeline, X0, y


if __name__ == '__main__':
    df = pd.read_csv(Path(config['input_path']))
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} cols")
    pipeline, X0, y = build_preprocessing_pipeline(df)
    X_trans = pipeline.fit_transform(X0)
    try:
        feature_names = pipeline.named_steps['prep'].get_feature_names_out()
    except Exception as e:
        logger.warning(f"Could not get feature names: {e}")
        feature_names = [f'feature_{i}' for i in range(X_trans.shape[1])]
    processed_df = pd.DataFrame(X_trans, columns=feature_names)
    processed_df[config['target_column']] = y.values
    processed_df.to_csv(config['output_path'], index=False)
    logger.info(f"Processed data saved to {config['output_path']}")
