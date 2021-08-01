import argparse
import os
import pickle
from copy import deepcopy

import numpy as np
import pytablewriter
import seaborn
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

parser = argparse.ArgumentParser("Gather results, plot them and create table")
parser.add_argument("-a", "--algos", help="Algorithms to include", nargs="+", type=str)
parser.add_argument("-e", "--env", help="Environments to include", nargs="+", type=str)
parser.add_argument("-f", "--exp-folders", help="Folders to include", nargs="+", type=str)
parser.add_argument("-l", "--labels", help="Label for each folder", nargs="+", type=str)
# parser.add_argument(
#     "-k",
#     "--key",
#     help="Key from the `evaluations.npz` file to use to aggregate results "
#     "(e.g. reward, success rate, ...), it is 'results' by default (i.e., the episode reward)",
#     default="results",
#     type=str,
# )
parser.add_argument("-max", "--max-timesteps", help="Max number of timesteps to display", type=int, default=int(2e6))
parser.add_argument("-min", "--min-timesteps", help="Min number of timesteps to keep a trial", type=int, default=-1)
parser.add_argument("-o", "--output", help="Output filename (pickle file), where to save the post-processed data", type=str)
parser.add_argument(
    "-median", "--median", action="store_true", default=False, help="Display median instead of mean in the table"
)
parser.add_argument("--no-million", action="store_true", default=False, help="Do not convert x-axis to million")
parser.add_argument("--no-display", action="store_true", default=False, help="Do not show the plots")
parser.add_argument(
    "-print", "--print-n-trials", action="store_true", default=False, help="Print the number of trial for each result"
)
parser.add_argument(
    "-exp", "--experiment-names", help="Separate different experiments by checking their suffix", nargs="+", type=str, default="", 
)
parser.add_argument("-t", "--tag", help="Specify tag from tensorboard here to plot it", default="eval/mean_reward", type=str)
args = parser.parse_args()

# Activate seaborn
seaborn.set()
results = {}
post_processed_results = {}

args.algos = [algo.upper() for algo in args.algos]

if args.labels is None:
    args.labels = args.exp_folders

for env in args.env:  # noqa: C901
    plt.figure(f"Results {env}")
    plt.title(f"{env}", fontsize=14)

    x_label_suffix = "" if args.no_million else "(in Million)"
    plt.xlabel(f"Timesteps {x_label_suffix}", fontsize=14)
    # plt.ylabel("Score", fontsize=14)
    plt.ylabel(f"{args.tag}", fontsize=14)
    results[env] = {}
    post_processed_results[env] = {}

    for algo in args.algos:
        for folder_idx, exp_folder in enumerate(args.exp_folders):
            for exp_idx, exp in enumerate(args.experiment_names):

                log_path = os.path.join(exp_folder, "tensorboard")

                if not os.path.isdir(log_path):
                    continue

                # What is this good for?
                # results[env][f"{args.labels[folder_idx]}-{algo}"] = 0.0

                for d in os.listdir(log_path):
                    if (env in d and os.path.isdir(os.path.join(log_path, d))):
                        log_path = os.path.join(log_path, d)
                        break

                if not(env in log_path):
                    continue 

                if algo == "MSAC" or algo == "M-SAC":
                    algo = "M-SAC"
                else:
                    exp = ""

                dirs = [
                    os.path.join(log_path, d)
                    for d in os.listdir(log_path)
                    if (d.startswith(algo) and (algo == "M-SAC" and d.endswith(exp) or exp_idx == 0 and d[-1].isdigit()) and os.path.isdir(os.path.join(log_path, d)))
                ]

                max_len = 0
                merged_timesteps, merged_results = [], []
                last_eval = []
                timesteps = np.empty(0)
                for _, dir_ in enumerate(dirs):
                    try:
                        log = EventAccumulator(dir_)
                        log.Reload()
                    except FileNotFoundError:
                        print("Data not found for", dir_)
                        continue
                    
                    # Somehow the last value is missing in the tensorboard values
                    # and the steps are odd compared to evaluations.npz
                    _, step_nums, mean_ = zip(*log.Scalars(args.tag))
                    mean_ = np.array(mean_)
                    step_nums = np.array(step_nums)

                    if mean_.shape == ():
                        continue

                    # For post-processing
                    df = pd.DataFrame({'value': mean_}, index=step_nums)
                    steps_resampled = np.arange(start=0, stop=args.max_timesteps, step=10000, dtype='int64')
                    df_resampled = df.reindex(df.index.union(steps_resampled)).interpolate(method='values').loc[steps_resampled]
                    step_nums = steps_resampled
                    mean_ = df_resampled.to_numpy().squeeze()
                    merged_timesteps.append(step_nums)
                    merged_results.append(mean_)

                    max_len = max(max_len, len(mean_))
                    if len(step_nums) >= max_len:
                        timesteps = step_nums

                    # Truncate the plots
                    # max_len = min(len(i) for i in merged_timesteps) # old approach to synchronize
                    while timesteps[max_len - 1] > args.max_timesteps:
                        max_len -= 1
                    timesteps = timesteps[:max_len]

                    if len(mean_) >= max_len:
                        last_eval.append(mean_[max_len - 1])
                    else:
                        last_eval.append(mean_[-1])

                # Remove incomplete runs
                # max_len = max(max_len, len(mean_))
                merged_results_tmp, last_eval_tmp = [], []
                for idx in range(len(merged_results)):
                    if len(merged_results[idx]) >= max_len:
                        merged_results_tmp.append(merged_results[idx][:max_len])
                        last_eval_tmp.append(last_eval[idx])
                merged_results = merged_results_tmp
                last_eval = last_eval_tmp

                # Post-process
                if len(merged_results) > 0:
                    # shape: (n_trials, n_eval * n_eval_episodes)
                    merged_results = np.array(merged_results)
                    n_trials = len(merged_results)
                    n_eval = len(timesteps)

                    if args.print_n_trials:
                        # print(f"{env}-{algo}-{args.labels[folder_idx]}: {n_trials}")
                        print(f"{env}_{algo}_{exp} in {args.labels[folder_idx]}: {n_trials}")

                    # reshape to (n_trials, n_eval, n_eval_episodes)
                    evaluations = merged_results.reshape((n_trials, n_eval, -1))
                    # re-arrange to (n_eval, n_trials, n_eval_episodes)
                    evaluations = np.swapaxes(evaluations, 0, 1)
                    # (n_eval,)
                    mean_ = np.mean(evaluations, axis=(1, 2))
                    # (n_eval, n_trials)
                    mean_per_eval = np.mean(evaluations, axis=-1)
                    # (n_eval,)
                    std_ = np.std(mean_per_eval, axis=-1)
                    # std: error:
                    std_error = std_ / np.sqrt(n_trials)
                    # Take last evaluation
                    last_evals = np.array(last_eval).squeeze()
                    # shape: (n_trials, n_eval_episodes) to (n_trials,)
                    if (type(last_eval[0]) is np.ndarray):
                        last_evals = last_evals.mean(axis=-1)
                    # Standard deviation of the mean performance for the last eval
                    std_last_eval = np.std(last_evals)
                    # Compute standard error
                    std_error_last_eval = std_last_eval / np.sqrt(n_trials)

                    if args.median:
                        # results[env][f"{algo}-{args.labels[folder_idx]}"] = f"{np.median(last_evals):.0f}"
                        results[env][f"{algo}_{exp}-{args.labels[folder_idx]}"] = f"{np.median(last_evals):.5f}"
                    else:
                        results[env][
                        #     f"{algo}-{args.labels[folder_idx]}"
                              f"{algo}_{exp}-{args.labels[folder_idx]}"
                        ] = f"{np.mean(last_evals):.5f} +/- {std_error_last_eval:.5f}"

                    # x axis in Millions of timesteps
                    divider = 1e6
                    if args.no_million:
                        divider = 1.0

                    # post_processed_results[env][f"{algo}-{args.labels[folder_idx]}"] = {
                    post_processed_results[env][f"{algo}_{exp}-{args.labels[folder_idx]}"] = {
                        "timesteps": timesteps,
                        "mean": mean_,
                        "std_error": std_error,
                        "last_evals": last_evals,
                        "std_error_last_eval": std_error_last_eval,
                    }

                    if "state_based" in exp:
                        plt.plot(timesteps / divider, mean_, label="$\mathregular{M-SAC_s}$, β = -10", linewidth=1.5)
                    elif "action_based" in exp:
                        plt.plot(timesteps / divider, mean_, label="$\mathregular{M-SAC_a}$, β = -10, τ = 0.01", linewidth=1.5)
                    else:
                        plt.plot(timesteps / divider, mean_, label=f"{algo}", linewidth=1.5)
                    # plt.plot(timesteps / divider, mean_, label=f"{algo}", linewidth=1.5)
                    plt.fill_between(timesteps / divider, mean_ + std_error, mean_ - std_error, alpha=0.3)

    plt.legend()


writer = pytablewriter.MarkdownTableWriter()
writer.table_name = "results_table"

headers = ["Environments"]

# One additional row for the subheader
value_matrix = [[] for i in range(len(args.env) + 1)]

value_matrix[0].append("")
for i, env in enumerate(args.env, start=1):
    value_matrix[i].append(env)
    for algo in args.algos:
        if algo == "MSAC":
            algo = "M-SAC"
        for label in args.labels:
            for exp in args.experiment_names:
                if algo == "M-SAC":
                    key = f"{algo}_{exp}-{label}"
                else:
                    key = f"{algo}_-{label}"
                    exp = ""
                if key in results[env] and f"{algo}_{exp}" not in headers:
                    if i == 1:
                        value_matrix[0].append(label)
                        headers.append(f"{algo}_{exp}")
                    value_matrix[i].append(f'{results[env][key]}')

writer.headers = headers
writer.value_matrix = value_matrix
writer.write_table()

post_processed_results["results_table"] = {"headers": headers, "value_matrix": value_matrix}

if args.output is not None:
    print(f"Saving to {args.output}.pkl")
    with open(f"{args.output}.pkl", "wb") as file_handler:
        pickle.dump(post_processed_results, file_handler)

if not args.no_display:
    plt.show()

"""
event_acc = EventAccumulator('/Users/Marcel/Repositories/tum-adlr-ss21-08/docs/results/archive/tensorboard/AntBulletEnv-v0/M-SAC_1')
event_acc.Reload()
# Show all tags in the log file
for key, value in event_acc.Tags().items():
    print(key)
    print(value)
# print(event_acc.Tags())

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
temp = event_acc.Scalars('eval/mean_reward')
w_times, step_nums, vals = zip(*event_acc.Scalars('eval/mean_reward'))
print("Hello!")
"""