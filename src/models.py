from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class LinearRegressionModel(BaseModel):
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

class RidgeModel(BaseModel):
    def __init__(self, alpha=1.0):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=alpha))
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

class LassoModel(BaseModel):
    def __init__(self, alpha=1.0):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=alpha))
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        # RF needs numeric input, so we use OrdinalEncoder for categorical features
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state,
            'n_jobs': -1
        }
        self.model = None

    def fit(self, X, y):
        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        num_cols = X.select_dtypes(exclude=['category', 'object']).columns.tolist()
        
        if cat_cols:
            # Preprocessing for RF: Ordinal Encode categoricals, pass through numerics
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
                    ('num', 'passthrough', num_cols)
                ]
            )
            self.model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(**self.params))
            ])
        else:
            self.model = RandomForestRegressor(**self.params)
            
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

class XGBoostModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        import xgboost as xgb
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'enable_categorical': True,
            'tree_method': 'hist', # Required for categorical support
            'n_jobs': -1
        }
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

class CatBoostModel(BaseModel):
    def __init__(self, iterations=100, depth=6, learning_rate=0.1, random_state=42):
        import catboost as cb
        self.params = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'random_seed': random_state,
            'verbose': 0,
            'allow_writing_files': False
        }
        self.model = cb.CatBoostRegressor(**self.params)

    def fit(self, X, y):
        # Identify categorical features
        cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
        self.model.fit(X, y, cat_features=cat_features)
        return self

    def predict(self, X):
        return self.model.predict(X)


from sklearn.linear_model import QuantileRegressor

class MedianRegressionModel(BaseModel):
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', QuantileRegressor(quantile=0.5, solver='highs'))
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
