# =============================================================================
# Author: Aniruddha Mukherjee
# RL Performance Plotting Script (Adapted for CartPole-v1)
#
# Description:
# This script loads reinforcement learning experiment data for DDQN, DQN, PPO,
# and RPI algorithms on the CartPole-v1 environment. It processes data
# over multiple seeds, calculates mean and standard deviation, and generates
# publication-quality plots and summary tables with enhanced styling.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from matplotlib.lines import Line2D

# =============================================================================
# --- 1. CONFIGURATION SECTION (Customize your plot here) ---
# =============================================================================

# --- Input/Output Settings ---
RESULTS_DIR = './results/net-arch/'
OUTPUT_DIR = os.path.join('plots', 'ICC', 'Cartpole', 'net-arch')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILENAME = os.path.join(
    OUTPUT_DIR, 'performance_comparison_cartpole_styled-FINAL-FINAL.png'
)
OUTPUT_FILENAME_PDF = os.path.join(
    OUTPUT_DIR, 'performance_comparison_cartpole_styled_SMALL.pdf'
)

# --- Experiment Settings ---
ALGORITHMS = ['DoubleDQN', 'PPO', 'RPI', 'DQN']
SEEDS = list(range(0, 10))

# --- Toggle to exclude specific architectures from plotting ---
EXCLUDED_ARCHS = []  # No exclusions

# --- Custom display names for plots/legends ---
ALGO_DISPLAY_NAMES = {
    'DoubleDQN': 'DoubleDQN',
    'PPO': 'PPO',
    'RPI': r'RPI$_{\mathrm{DQN}}$',
    'DQN': 'DQN'
}

# --- Data file names (as per the directory structure) ---
TRUE_RETURN_FILE = 'mc_disc_returns.npy'
ESTIMATED_VALUE_FILE = 'v_s0_estimates.npy'
TIMESTEPS_FILE = 'time_steps_of_eval.npy'

# --- Plotting Style & Aesthetics ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "figure.titlesize": 14
})

# --- Colors for the algorithms ---
ALGO_COLORS = {
    'DoubleDQN': "#095384",  # blue
    'PPO': '#7570B3',        # purple
    'DQN': '#D95F02',      # orange
    'RPI': '#1B9E77',      # green
}

# Plot layout
SUBPLOT_COLS = 3

# --- Sparsity Settings ---
SPARSITY = 10  # Plot every Nth point (set to 1 to plot all points)


# =============================================================================
# --- 2. HELPER FUNCTIONS (Data loading and processing) ---
# =============================================================================

def natural_sort_key(s):
    """Creates a key for sorting strings containing numbers in a natural order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def rowwise_mean(arr):
    """Takes a numpy array and robustly flattens it to 1D."""
    if arr.ndim > 1:
        return arr.reshape(arr.shape[0], -1).mean(axis=1)
    return arr


def process_experiment_data(base_path, algorithm, architecture, seeds):
    """Loads and processes data for a single experiment configuration."""

    print(
        f"Entered `process_experiment_data` with process_experiment_data({base_path}, {algorithm}, {architecture}, {seeds})")

    true_return_curves, estimated_value_curves, all_timesteps = [], [], []

    for seed in seeds:
        exp_dir = os.path.join(base_path, algorithm,
                               str(architecture), f'seed_{seed}')
        if not os.path.isdir(exp_dir):
            if len(seeds) < 10:
                print(f"Warning – missing run: {exp_dir}")
            continue

        try:
            true_return_raw = np.load(os.path.join(
                exp_dir, TRUE_RETURN_FILE), allow_pickle=True)
            # print(f"true_return_raw.shape: {true_return_raw.shape}")
            true_returns = rowwise_mean(np.asarray(
                true_return_raw, dtype=np.float64))
            # print(f"true_returns.shape: {true_returns.shape}")

            est_value_raw = np.load(os.path.join(
                exp_dir, ESTIMATED_VALUE_FILE), allow_pickle=True)
            # print(f"est_value_raw.shape: {est_value_raw.shape}")
            est_values = rowwise_mean(np.asarray(
                est_value_raw, dtype=np.float64))
            # print(f"est_values.shape: {est_values.shape}")

            steps_raw = np.load(os.path.join(
                exp_dir, TIMESTEPS_FILE), allow_pickle=True)
            # print(f"steps_raw.shape: {steps_raw.shape}")
            steps = rowwise_mean(np.asarray(steps_raw, dtype=np.int64))
            # print(f"steps.shape: {steps.shape}")

            # --- MODIFICATION: Scale PPO timesteps by a factor of 8 ---
            if algorithm == 'PPO':
                steps = steps * 8

        except FileNotFoundError as e:
            print(f"Warning: File not found in {exp_dir} - {e}")
            continue

        true_return_curves.append(true_returns)
        estimated_value_curves.append(est_values)
        all_timesteps.append(steps)

    if not true_return_curves:
        return None, None, None, None

    # print(f"true_return_curves: {true_return_curves}")
    print(f"len(true_return_curves): {len(true_return_curves)}")
    # print(f"estimated_value_curves: {estimated_value_curves}")

    min_len = min(len(c) for c in true_return_curves)

    # print(f"printing len(c) for each of the curves in true_return_curves")
    # for c in true_return_curves:
    #     print(len(c))

    # true_return_curves = [c[:min_len] for c in true_return_curves]
    # estimated_value_curves = [c[:min_len] for c in estimated_value_curves]
    # all_timesteps = [c[:min_len] for c in all_timesteps]

    true_return_stack = np.stack(true_return_curves)
    est_value_stack = np.stack(estimated_value_curves)
    timesteps_stack = np.stack(all_timesteps)

    # print(true_return_stack)

    true_return_mean = true_return_stack.mean(axis=0)
    true_return_std = true_return_stack.std(axis=0)
    est_value_mean = est_value_stack.mean(axis=0)
    timesteps = timesteps_stack.mean(axis=0)

    return timesteps, true_return_mean, true_return_std, est_value_mean

# =============================================================================
# --- 3. MAIN PLOTTING FUNCTION ---
# =============================================================================


def generate_plots(arch_dirs):
    """Orchestrates the data loading, processing, and plotting for all experiments."""
    num_archs = len(arch_dirs)
    if num_archs == 0:
        print("No architectures to plot. Aborting.")
        return

    subplot_rows = int(np.ceil(num_archs / SUBPLOT_COLS))
    fig, axes = plt.subplots(
        subplot_rows, SUBPLOT_COLS,
        figsize=(SUBPLOT_COLS * 4.5, subplot_rows * 2.8),
        sharex=True, sharey=True,
    )
    axes = axes.flatten() if num_archs > 1 else [axes]

    for i, arch_name in enumerate(arch_dirs):
        ax = axes[i]
        for algo in ALGORITHMS:
            timesteps, true_return_mean, true_return_std, est_value_mean = process_experiment_data(
                RESULTS_DIR, algo, arch_name, SEEDS
            )
            if timesteps is None or len(timesteps) == 0:
                continue

            color = ALGO_COLORS.get(algo, 'gray')
            display_name = ALGO_DISPLAY_NAMES.get(algo, algo)

            ax.plot(timesteps[::SPARSITY], true_return_mean[::SPARSITY], color=color,
                    linestyle='-', label=f'{display_name} Return', linewidth=3)
            ax.fill_between(timesteps[::SPARSITY], true_return_mean[::SPARSITY] - true_return_std[::SPARSITY],
                            true_return_mean[::SPARSITY] + true_return_std[::SPARSITY], color=color, alpha=0.2)
            ax.plot(timesteps[::SPARSITY], est_value_mean[::SPARSITY], color=color,
                    linestyle='--', label=f'{display_name} Value Estimate', linewidth=2)

        # --- MODIFICATION: Updated plot aesthetics for CartPole with new PPO scale ---
        # Extract width from arch_name (e.g., "width_128_depth_2" -> "128-128")
        match = re.search(r'width_(\d+)_depth_(\d+)', arch_name)
        if match:
            width = match.group(1)
            arch_label = f'{width}-{width}'
        else:
            arch_label = arch_name

        ax.set_title(f'{arch_label}')
        ax.grid(True, which='both', linestyle='--', linewidth=1.5)
        ax.set_ylim(0, 125)  # Adjusted for CartPole's typical return values
        ax.set_xlim(0, 1e5)  # Adjusted for CartPole's typical return values
        ax.set_xticks([0, 50000, 100000])
        ax.set_xticklabels(['0', '0.5', '1'])

        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(['0', '50', '100'])

    for i in range(num_archs, len(axes)):
        fig.delaxes(axes[i])

    # Dynamically create legend elements for active algorithms
    legend_elements = []
    for algo in ALGORITHMS:
        if algo in ALGO_COLORS:
            legend_elements.append(
                Line2D([0], [0], color=ALGO_COLORS[algo],
                       lw=3, label=ALGO_DISPLAY_NAMES[algo])
            )

    ncol = min(len(legend_elements), 5)
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=ncol, bbox_to_anchor=(0.5, 1))

    fig.text(0.5, 0.01, r'Environment Timesteps ($\times 10^5$)',
             ha='center', va='center', fontsize=18)
    fig.text(0.06, 0.5, 'Return', ha='center',
             va='center', rotation='vertical', fontsize=18)

    plt.tight_layout(
        rect=[0.08, 0.05, 0.98, 0.92],  # [left, bottom, right, top] margins
        pad=1,      # Padding between figure edge and subplots (default: 1.08)
        h_pad=0.2,    # Height padding between subplots (default: 1.08)
        w_pad=0.2     # Width padding between subplots (default: 1.08)
    )
    plt.savefig(OUTPUT_FILENAME, dpi=250, bbox_inches='tight')
    plt.savefig(OUTPUT_FILENAME_PDF, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully generated and saved to '{OUTPUT_FILENAME}'")
    plt.show()

# =============================================================================
# --- 4. TABLE GENERATION FUNCTION ---
# =============================================================================


def generate_table_data(arch_dirs):
    """Calculates aggregate metrics and generates a pandas DataFrame for CartPole."""
    if not arch_dirs:
        print("No architectures to process for the table after exclusions.")
        return

    print("\n--- Generating Table Data ---")
    table_data = []

    for arch_name in arch_dirs:
        row_data = {'Net-Arch': arch_name}
        for algo in ALGORITHMS:
            final_perf_per_seed, aucs_per_seed = [], []

            for seed in SEEDS:
                # The function call below now correctly returns scaled timesteps for PPO
                timesteps, true_return_seed, _, _ = process_experiment_data(
                    RESULTS_DIR, algo, arch_name, [seed]
                )

                if timesteps is not None and len(timesteps) > 0 and true_return_seed is not None and len(true_return_seed) > 0:
                    num_final_points = max(1, int(len(true_return_seed) * 0.1))
                    final_perf_per_seed.append(
                        np.mean(true_return_seed[-num_final_points:]))
                    # AUC calculation correctly uses the scaled timesteps
                    # if algo=='PPO':
                    #     print(timesteps)
                    #     break
                    aucs_per_seed.append(np.trapz(true_return_seed, timesteps))

            if final_perf_per_seed:
                final_perf_mean, final_perf_std = np.mean(
                    final_perf_per_seed), np.std(final_perf_per_seed)
                auc_mean, auc_std = np.mean(
                    aucs_per_seed), np.std(aucs_per_seed)

                display_name = ALGO_DISPLAY_NAMES.get(algo, algo)
                row_data[f'{display_name} Final Perf.'] = f"{final_perf_mean:.2f} ± {final_perf_std:.2f}"
                row_data[f'{display_name} AUC'] = f"{auc_mean:.0f} ± {auc_std:.0f}"
            else:
                display_name = ALGO_DISPLAY_NAMES.get(algo, algo)
                row_data[f'{display_name} Final Perf.'] = 'N/A'
                row_data[f'{display_name} AUC'] = 'N/A'

        table_data.append(row_data)

    if not table_data:
        print("No data was processed successfully. Cannot generate table.")
        return

    df = pd.DataFrame(table_data).set_index('Net-Arch')
    print("\n--- Results Table (CartPole) ---")
    print(df.to_string())

    csv_filename = os.path.join(OUTPUT_DIR, 'results_summary_cartpole.csv')
    df.to_csv(csv_filename)
    print(f"\nTable saved to {csv_filename}")

    latex_filename = os.path.join(OUTPUT_DIR, 'results_summary_cartpole.tex')
    df.to_latex(latex_filename, longtable=False)
    print(f"Table saved to {latex_filename}")

# =============================================================================
# --- 5. SCRIPT EXECUTION ---
# =============================================================================


if __name__ == '__main__':
    try:
        if not ALGORITHMS:
            print("FATAL: The 'ALGORITHMS' list in the configuration is empty.")
            exit()

        # Discover architectures from the first algorithm's directory
        arch_discovery_path = os.path.join(RESULTS_DIR, ALGORITHMS[0])
        print(f"arch_discovery_path: {arch_discovery_path}")
        if not os.path.isdir(arch_discovery_path):
            print(
                f"FATAL: The directory for the first algorithm '{arch_discovery_path}' was not found.")
            exit()

        all_dirs = os.listdir(arch_discovery_path)

        print(f"all_dirs: {all_dirs}")
        # Ensure we only process actual directories and ignore files like .DS_Store
        arch_dirs_to_process = sorted(
            [d for d in all_dirs if os.path.isdir(os.path.join(
                arch_discovery_path, d)) and d not in EXCLUDED_ARCHS],
            key=natural_sort_key
        )

        print(f"arch_dirs_to_process: {arch_dirs_to_process}")

    except FileNotFoundError:
        print(f"FATAL: The results directory '{RESULTS_DIR}' was not found.")
        exit()

    if not arch_dirs_to_process:
        print(
            f"FATAL: No plottable architecture directories found in '{arch_discovery_path}' (after exclusions).")
        exit()

    print(
        f"Found {len(arch_dirs_to_process)} architectures to process after exclusions.")
    if EXCLUDED_ARCHS:
        print(f"Excluded architectures: {', '.join(EXCLUDED_ARCHS)}")

    generate_plots(arch_dirs_to_process)
    generate_table_data(arch_dirs_to_process)
