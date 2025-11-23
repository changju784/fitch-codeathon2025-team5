import os
import json
import datetime
import shutil
import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class ExperimentManager:
    def __init__(self, experiment_dir="experiments/logs"):
        self.experiment_dir = experiment_dir
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.current_exp_dir = None
        self.config = None

    def create_experiment(self, config, name=None):
        """
        Creates a new experiment directory and saves the config.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = name if name else config.get("name", "experiment")
        dir_name = f"{timestamp}_{exp_name}"
        self.current_exp_dir = os.path.join(self.experiment_dir, dir_name)
        os.makedirs(self.current_exp_dir, exist_ok=True)
        
        self.config = config
        self._save_json("config.json", config)
        print(f"Experiment created at: {self.current_exp_dir}")
        
        # Setup logging
        log_file = os.path.join(self.current_exp_dir, "train.log")
        sys.stdout = Logger(log_file)
        print(f"Logging to {log_file}")
        
        return self.current_exp_dir

    def load_experiment(self, experiment_id):
        """
        Loads config from an existing experiment directory.
        experiment_id can be the full path or just the folder name.
        """
        if os.path.exists(experiment_id):
            self.current_exp_dir = experiment_id
        else:
            self.current_exp_dir = os.path.join(self.experiment_dir, experiment_id)
            
        if not os.path.exists(self.current_exp_dir):
            raise ValueError(f"Experiment {experiment_id} not found.")
            
        config_path = os.path.join(self.current_exp_dir, "config.json")
        if not os.path.exists(config_path):
             raise ValueError(f"Config file not found in {self.current_exp_dir}")
             
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        print(f"Loaded experiment from: {self.current_exp_dir}")
        
        # Setup logging for loaded experiment too (appending)
        log_file = os.path.join(self.current_exp_dir, "train.log")
        sys.stdout = Logger(log_file)
        print(f"Logging to {log_file}")
        
        return self.config

    def log_metrics(self, metrics, filename="metrics.json"):
        """
        Saves metrics to the experiment directory.
        """
        if not self.current_exp_dir:
            raise RuntimeError("No active experiment. Call create_experiment or load_experiment first.")
        self._save_json(filename, metrics)
        print(f"Metrics saved to {filename}")

    def _save_json(self, filename, data):
        path = os.path.join(self.current_exp_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
