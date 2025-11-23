import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Trainer:
    def __init__(self, model_class, model_params=None, n_splits=5, random_state=42, log_target=False):
        self.model_class = model_class
        self.model_params = model_params if model_params else {}
        self.n_splits = n_splits
        self.random_state = random_state
        self.log_target = log_target
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        return {"rmse": rmse, "mse": mse, "mae": mae}

    def train(self, X, y):
        fold_metrics = []
        top_errors = []
        
        print(f"Starting training with {self.n_splits} folds... (Log Target: {self.log_target})")
        
        # Reset index of y if it's a series to ensure alignment if X is dataframe
        if hasattr(y, 'reset_index'):
            y = y.reset_index(drop=True)
        if hasattr(X, 'reset_index'):
            X = X.reset_index(drop=True)
            
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X, y)):
            # Handle DataFrame or numpy array indexing
            if hasattr(X, 'iloc'):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
                
            if hasattr(y, 'iloc'):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            
            # Apply log transform to target if requested
            y_train_processed = np.log1p(y_train) if self.log_target else y_train
            
            # Initialize model
            model = self.model_class(**self.model_params)
            
            # Train
            model.fit(X_train, y_train_processed)
            
            # Predict
            y_pred_processed = model.predict(X_val)
            
            # Inverse transform prediction if needed
            y_pred = np.expm1(y_pred_processed) if self.log_target else y_pred_processed
            
            # Evaluate
            metrics = self.evaluate(y_val, y_pred)
            
            # Log-scale metrics (Always calculate)
            # Clip predictions to 0 to avoid log(negative)
            y_pred_clipped = np.maximum(y_pred, 0)
            y_val_clipped = np.maximum(y_val, 0)
            
            y_val_log = np.log1p(y_val_clipped)
            y_pred_log = np.log1p(y_pred_clipped)
            
            metrics_log = self.evaluate(y_val_log, y_pred_log)
            metrics.update({f"log_{k}": v for k, v in metrics_log.items()})
            
            fold_metrics.append(metrics)
            print(f"Fold {fold+1}/{self.n_splits} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f} | Log RMSE: {metrics['log_rmse']:.4f}, Log MAE: {metrics['log_mae']:.4f}")
                
            # Calculate Top 5 Errors
            # Calculate absolute error
            abs_errors = np.abs(y_val - y_pred)
            # Get indices of top 5 errors
            # If y_val is series, we need to preserve original indices or handle carefully
            # Here y_val is a slice, so its index is preserved from original y
            
            if hasattr(y_val, 'index'):
                error_df = pd.DataFrame({
                    'actual': y_val,
                    'predicted': y_pred,
                    'abs_error': abs_errors
                }, index=y_val.index)
            else:
                error_df = pd.DataFrame({
                    'actual': y_val,
                    'predicted': y_pred,
                    'abs_error': abs_errors
                })
                
            top_5_fold = error_df.nlargest(5, 'abs_error')
            top_errors.append(top_5_fold)

        # Aggregate metrics
        agg_metrics = {}
        for key in fold_metrics[0].keys():
            values = [m[key] for m in fold_metrics]
            agg_metrics[f"mean_{key}"] = np.mean(values)
            agg_metrics[f"std_{key}"] = np.std(values)
        
        print(f"\nTraining Completed.")
        print(f"Mean RMSE: {agg_metrics['mean_rmse']:.4f}")
        print(f"Mean MAE: {agg_metrics['mean_mae']:.4f}")
        print(f"Mean Log RMSE: {agg_metrics['mean_log_rmse']:.4f}")
        print(f"Mean Log MAE: {agg_metrics['mean_log_mae']:.4f}")
            
        # Aggregate Top Errors
        all_top_errors = pd.concat(top_errors).nlargest(5, 'abs_error')
        print("\nTop 5 Worst Predictions (Across all folds):")
        print(all_top_errors)
        
        # Add top errors to metrics for logging (convert to dict)
        agg_metrics['top_5_errors'] = all_top_errors.to_dict(orient='index')
        
        return agg_metrics
