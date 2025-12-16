import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


def create_objective(args):
    def objective(trial: optuna.Trial) -> float:
        hyperparams = {
            "trial_number": trial.number,
            "lr_actor": trial.suggest_float("lr_actor", 1e-5, 1e-2, log=True),
            "lr_critic": trial.suggest_float("lr_critic", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "tau": trial.suggest_float("tau", 0.001, 0.05, log=True),
            "policy_delay": trial.suggest_int("policy_delay", 1, 4),
            "noise_std": trial.suggest_float("noise_std", 0.05, 0.3),
            "noise_clip": trial.suggest_float("noise_clip", 0.1, 0.5),
        }

        logging.info(f"Trial {trial.number} - Hyperparams: {hyperparams}")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as hp_file:
            json.dump(hyperparams, hp_file)
            hp_path = hp_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as out_file:
            out_path = out_file.name

        try:
            cmd = [
                sys.executable, "-m", "ACC.Agents.HyperParamTrailRunner",
                "--hyperparams-file", hp_path,
                "--output-file", out_path,
                "--trial-duration", str(args.trial_duration),
            ]

            if args.verbose:
                cmd.append("--verbose")

            logging.info(f"Launching trial subprocess...")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            for line in process.stdout:
                if '%|' in line or 'it/s]' in line:
                    continue
                print(line, end='')

            return_code = process.wait()

            logging.info(f"Trial subprocess finished with code {return_code}")

            if os.path.exists(out_path):
                with open(out_path, 'r') as f:
                    result = json.load(f)

                if result.get("success"):
                    avg_reward = result["avg_reward"]
                    logging.info(f"Trial {trial.number} completed. Avg Reward: {avg_reward:.4f}")
                    return avg_reward
                else:
                    logging.error(f"Trial {trial.number} failed: {result.get('error')}")
                    return float('-inf')
            else:
                logging.error(f"Trial {trial.number} - No output file found")
                return float('-inf')

        finally:
            for path in [hp_path, out_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except:
                    pass

            # Extra cleanup time between trials
            time.sleep(5)

    return objective


def run_hyperparameter_search(args):
    """Run the Bayesian hyperparameter search."""

    sampler = TPESampler(n_startup_trials=args.n_startup_trials, seed=args.seed)
    pruner = MedianPruner(n_startup_trials=args.n_startup_trials)

    storage = f"sqlite:///{args.study_name}.db" if args.persist_study else None

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=args.persist_study
    )

    objective = create_objective(args)

    logging.info(f"Starting Bayesian HPO with {args.n_trials} trials...")
    logging.info(f"Each trial trains for {args.trial_duration} seconds")

    try:
        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    except KeyboardInterrupt:
        logging.info("Optimization interrupted by user.")

    # Print results
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("=" * 60)

    if len(study.trials) > 0:
        print(f"\nBest Trial: {study.best_trial.number}")
        print(f"Best Avg Reward: {study.best_value:.4f}")
        print("\nBest Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Save results
        study.trials_dataframe().to_csv(f"{args.study_name}_results.csv", index=False)
        print(f"\nResults saved to {args.study_name}_results.csv")

        if args.visualize:
            try:
                import optuna.visualization as vis
                vis.plot_optimization_history(study).write_html(f"{args.study_name}_history.html")
                vis.plot_param_importances(study).write_html(f"{args.study_name}_importance.html")
                print("Visualizations saved as HTML files.")
            except ImportError:
                logging.warning("Install plotly for visualizations: pip install plotly")
    else:
        print("No trials completed successfully.")

    return study


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian HPO for CARLA ACC TD3')

    # HPO settings
    parser.add_argument('--n-trials', default=50, type=int)
    parser.add_argument('--n-startup-trials', default=10, type=int)
    parser.add_argument('--trial-duration', default=30*60, type=int, help='Seconds per trial')
    parser.add_argument('--timeout', default=None, type=int, help='Total timeout in seconds')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--study-name', default='td3_hpo', type=str)
    parser.add_argument('--persist-study', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 60)
    print("TD3 BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Trials: {args.n_trials}")
    print(f"Trial Duration: {args.trial_duration // 60} minutes each")
    print(f"Running each trial in isolated subprocess")
    print("=" * 60 + "\n")

    run_hyperparameter_search(args)





    study = optuna.load_study(study_name="td3_hpo", storage="sqlite:///td3_hpo.db")
    df = study.trials_dataframe()
    print(df[['number', 'value', 'state']])