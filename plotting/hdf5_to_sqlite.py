import h5py
import yaml
import logging
import numpy as np
from datetime import datetime
import os
import json
import hashlib
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_hash_cache(cache_file):
    """Load the hash cache from JSON file."""
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_hash_cache(cache_file, cache):
    """Save the hash cache to JSON file."""
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

def create_tables(conn):
    """Create necessary database tables if they don't exist."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT,
            timestamp DATETIME,
            metric_name TEXT,
            value REAL,
            metric_type TEXT
        )
    ''')
    
    # Create an index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
        ON metrics(timestamp)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_metrics_run_name 
        ON metrics(run_name)
    ''')
    
    conn.commit()

def process_experiment(experiment_path, conn, hash_cache, cache_file):
    """Process a single experiment directory and write its data to SQLite."""
    # Get the actual experiment directory (the timestamped folder)
    experiment_dirs = [d for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]
    if not experiment_dirs:
        logging.warning(f"No experiment directories found in {experiment_path}")
        return
    
    working_dir = os.path.join(experiment_path, experiment_dirs[0])
    log_file = os.path.join(working_dir, "experiment_dir/logs/log_no_seed_provided.hdf5")
    hydra_yaml = os.path.join(working_dir, ".hydra/hydra.yaml")
    config_yaml = os.path.join(working_dir, ".hydra/config.yaml")

    # Skip if required files don't exist
    if not all(os.path.exists(f) for f in [log_file, config_yaml]):
        logging.warning(f"Missing required files in {working_dir}")
        return

    # Calculate current file hash
    current_hash = calculate_file_hash(log_file)

    with open(hydra_yaml, "r") as f:
        hydra_config = yaml.safe_load(f)

    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)
        num_train_iterations = config["agent"]["policy"]["train_iterations"]
        run_name = config["job"]["name"]
        if run_name == "default_run":
            run_name = hydra_config["hydra"]["job"]["name"]

    # Check if hash has changed
    if run_name in hash_cache and hash_cache[run_name] == current_hash:
        logging.info(f"No changes detected for {run_name}, skipping...")
        return

    cursor = conn.cursor()
    
    try:
        with h5py.File(log_file, "r") as f:
            data = f["no_seed_provided"]
            stats = data["stats"]
            time = data["time"]
            
            timestamps = time["time"][:]
            timestamps = [timestamp.decode("utf-8") for timestamp in timestamps]
            timestamp_indices = np.arange(len(timestamps))
            
            iteration_mask = ((timestamp_indices-(num_train_iterations)) % (num_train_iterations + 1) == 0)
            
            iteration_counter = 0
            step_counter = 0
            
            # Prepare batch inserts
            records = []
            
            for idx, timestamp in enumerate(timestamps):
                timestamp = datetime.strptime(timestamp, "%y-%m-%d/%H:%M")

                if iteration_mask[idx]:
                    fields = {
                        "num_iterations": time["num_iterations"][iteration_counter],
                        "val_loss": stats["val_loss"][iteration_counter],
                        "final_goals_proven": stats["final_goals_proven"][iteration_counter],
                        "ratio_proven": stats["ratio_proven"][iteration_counter],
                        "mean_hard_sol_log_probs": stats["mean_hard_sol_log_probs"][iteration_counter]
                    }
                    metric_type = 'iteration'
                    iteration_counter += 1
                else:
                    fields = {
                        "num_steps": time["num_steps"][step_counter],
                        "loss": stats["loss"][step_counter],
                        "train_loss": stats["train_loss"][step_counter],
                        "progress_loss": stats["progress_loss"][step_counter],
                        "mu": stats["mu"][step_counter],
                        "ratio_diff_problem_pairs": stats["ratio_diff_problem_pairs"][step_counter]
                    }
                    metric_type = 'step'
                    step_counter += 1

                for metric, value in fields.items():
                    try:
                        value = float(value)
                        if not (np.isnan(value) or np.isinf(value)):
                            records.append((
                                run_name,
                                timestamp,
                                metric,
                                value,
                                metric_type
                            ))
                    except (ValueError, TypeError):
                        logging.warning(f"Error processing metric {metric} with value {value}")
                        continue

            # Batch insert records
            if records:
                cursor.executemany(
                    """
                    INSERT INTO metrics (run_name, timestamp, metric_name, value, metric_type)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    records
                )
                conn.commit()
                logging.info(f"Inserted {len(records)} records for {run_name}")

        # Update hash cache after successful processing
        hash_cache[run_name] = current_hash
        save_hash_cache(cache_file, hash_cache)
        logging.info(f"Updated hash cache for {run_name}")

    except Exception as e:
        conn.rollback()
        logging.error(f"Error processing experiment {run_name}: {str(e)}")
        raise

def main():
    # Create database directory if it doesn't exist
    db_dir = Path(os.path.dirname(__file__)) / "db"
    db_dir.mkdir(exist_ok=True)
    
    # Configure SQLite connection
    db_path = db_dir / "experiments.db"
    conn = sqlite3.connect(str(db_path))
    
    # Create tables if they don't exist
    create_tables(conn)

    # Initialize hash cache
    cache_file = os.path.join(os.path.dirname(__file__), 'hdf5_hashes.json')
    hash_cache = load_hash_cache(cache_file)

    base_dir = os.getenv("EXPERIMENTS_DIR")
    
    try:
        for experiment_name in os.listdir(base_dir):
            experiment_path = os.path.join(base_dir, experiment_name)
            if os.path.isdir(experiment_path):
                logging.info(f"Processing experiment: {experiment_name}")
                try:
                    process_experiment(experiment_path, conn, hash_cache, cache_file)
                except Exception as e:
                    logging.error(f"Error processing {experiment_name}: {str(e)}")
                    continue
    except Exception as e:
        logging.error(f"Error processing experiments: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()