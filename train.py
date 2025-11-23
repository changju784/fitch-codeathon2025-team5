import pandas as pd
import argparse
import json
import sys
import os
import src.models as models_module
import src.preprocessor as preprocessor_module
from src.trainer import Trainer
from src.experiment_manager import ExperimentManager

def load_data():
    print("Loading data...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    return train, test

def get_default_config():
    return {
        "name": "baseline_run",
        "model": {
            "name": "LinearRegressionModel",
            "params": {}
        },
        "preprocessor": {
            "name": "CodeathonBaselinePreprocessor",
            "params": {}
        },
        "trainer": {
            "n_splits": 5,
            "random_state": 42
        }
    }

def get_class(module, class_name):
    if hasattr(module, class_name):
        return getattr(module, class_name)
    else:
        raise ValueError(f"Class {class_name} not found in module {module.__name__}")

def main():
    parser = argparse.ArgumentParser(description="Train Fitch Codeathon Model")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--exp-id", type=str, help="Experiment ID to load config from")
    args = parser.parse_args()

    manager = ExperimentManager()
    
    if args.exp_id:
        config = manager.load_experiment(args.exp_id)
        manager.create_experiment(config, name=f"rerun_{config.get('name', 'exp')}")
    elif args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        manager.create_experiment(config)
    else:
        print("No config provided. Using default baseline config.")
        config = get_default_config()
        manager.create_experiment(config)

    train_df, test_df = load_data()
    
    # Initialize Preprocessor dynamically
    preprocessor_config = config.get("preprocessor", {})
    preprocessor_class_name = preprocessor_config.get("name", "BaselinePreprocessor")
    preprocessor_params = preprocessor_config.get("params", {})
    
    print(f"Initializing preprocessor: {preprocessor_class_name}")
    PreprocessorClass = get_class(preprocessor_module, preprocessor_class_name)
    preprocessor = PreprocessorClass(**preprocessor_params)
    
    print("Preprocessing data...")
    # Fit on train and transform train
    X = preprocessor.fit_transform(train_df)
    
    # Transform test
    X_test = preprocessor.transform(test_df)
    
    # Extract and log selected features
    selected_features = []
    if hasattr(preprocessor, "selected_features"):
        selected_features = preprocessor.selected_features
    elif hasattr(preprocessor, "feature_names"):
        selected_features = preprocessor.feature_names
        
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    
    # Save features to file
    manager._save_json("features.json", {"features": selected_features})
    
    # Target extraction
    y_scope1 = train_df['target_scope_1']
    y_scope2 = train_df['target_scope_2']
    
    # Initialize Trainer with dynamic model
    model_config = config.get("model", {})
    model_class_name = model_config.get("name", "LinearRegressionModel")
    model_params = model_config.get("params", {})
    
    print(f"Initializing model: {model_class_name}")
    ModelClass = get_class(models_module, model_class_name)
    
    trainer_params = config.get("trainer", {})
    trainer = Trainer(
        model_class=ModelClass,
        model_params=model_params,
        **trainer_params
    )
    
    metrics = {}
    
    # 1. Cross-Validation
    # 1. Cross-Validation
    print("\n--- Training Target Scope 1 (CV) ---")
    metrics["scope_1"] = trainer.train(X, y_scope1)
    
    print("\n--- Training Target Scope 2 (CV) ---")
    metrics["scope_2"] = trainer.train(X, y_scope2)
    
    manager.log_metrics(metrics)
    
    # 2. Full Training and Submission
    print("\n--- Training Final Models and Generating Submission ---")
    
    # Train Scope 1 on full data
    model_s1 = ModelClass(**model_params)
    model_s1.fit(X, y_scope1)
    s1_predictions = model_s1.predict(X_test)
    
    # Train Scope 2 on full data
    model_s2 = ModelClass(**model_params)
    model_s2.fit(X, y_scope2)
    s2_predictions = model_s2.predict(X_test)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'entity_id': test_df['entity_id'],
        's1_predictions': s1_predictions,
        's2_predictions': s2_predictions
    })
    
    # Save submission
    submission_path = os.path.join(manager.current_exp_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    main()
