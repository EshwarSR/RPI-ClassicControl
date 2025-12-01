# =============================================================================
#
# RL Performance Plotting Script (InvertedPendulum-v5)
#
# Description:
# This script loads reinforcement learning experiment data for DDPG, DDPG_RPI,
# PPO, and TD3 algorithms on the InvertedPendulum-v5 environment. It processes
# data over multiple seeds, calculates mean and standard deviation, and
# generates publication-quality plots and summary tables.
#
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
RESULTS_DIR = './results/net-arch/InvertedPendulum-v5/'
PPO_RESULTS_DIR = './Net-Arch-Exp_PPO/PPO_Runs/results/net-arch/InvertedPendulum-v5/'
PPO_RESULTS_DIR_2 = './Net-Arch-Exp_PPO/PPO_Runs/results/net-arch/InvertedPendulum-v5/'
OUTPUT_FILENAME = 'performance_comparison_inverted_pendulum_styled-FINAL-FINAL.png'
OUTPUT_FILENAME_PDF = 'performance_comparison_inverted_pendulum_styled-PDF.pdf'

# --- Experiment Settings ---

ALGORITHMS = [
            'DDPG',
            'DDPG_RPI',
            # 'PPO',
            'PPO_0.999',
            'TD3'
                ]

SEEDS = list(range(0, 10))

# --- Toggle to exclude specific architectures from plotting ---
EXCLUDED_ARCHS = [
    # 'arch_1024-1024_c_2.0_lambda1_10.0_lambda2_2.0_rmin_1.0'
]

# --- Custom display names for plots/legends ---
ALGO_DISPLAY_NAMES = {
    'DDPG': 'DDPG',
    'DDPG_RPI': r'RPI$_{\mathrm{DDPG}}$',
    # 'PPO': 'PPO',
    'PPO_0.999': r'PPO',
    'TD3': 'TD3'
}

# --- Data file names ---
TRUE_Q_FILE_DDPG = 'true_q_vals.npy'
Q_HAT_FILE_DDPG = 'q_hat_vals.npy'
TIMESTEPS_FILE_DDPG = 'time_steps.npy'

TRUE_Q_FILE_PPO = 'mc_disc_returns.npy'
Q_HAT_FILE_PPO = 'v_s0_estimates.npy'
TIMESTEPS_FILE_PPO = 'time_steps_of_eval.npy'

# --- MODIFICATION: Updated Plotting Style & Aesthetics ---
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

# --- Colors for the algorithms (using colors from Code1) ---
ALGO_COLORS = {
    'DDPG':     '#D95F02',
    'DDPG_RPI': '#1B9E77',
    # 'PPO':      '#7570B3', #FF69B4
    'PPO_0.999': '#7570B3',  # Pink color
    'TD3':      '#095384'
}

# Plot layout
SUBPLOT_COLS = 3

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
    true_q_curves, q_hat_curves, all_timesteps = [], [], []

    for seed in seeds:
        # Use PPO-specific directory if algorithm is PPO or PPO_0.999
        # PPO uses seed_1 through seed_10, so we add 1 to the seed index
        if algorithm == 'PPO':
            exp_dir = os.path.join(PPO_RESULTS_DIR, 'PPO', architecture, f"seed_{seed+1}")
        elif algorithm == 'PPO_0.999':
            exp_dir = os.path.join(PPO_RESULTS_DIR_2, 'PPO', architecture, f"seed_{seed+1}")
        else:
            exp_dir = os.path.join(base_path, algorithm, architecture, str(seed))

        if not os.path.isdir(exp_dir):
            if len(seeds) < 10: print(f"Warning – missing run: {exp_dir}")
            continue

        try:
            if algorithm in ['PPO', 'PPO_0.999']:
                true_q_file, q_hat_file, timesteps_file = TRUE_Q_FILE_PPO, Q_HAT_FILE_PPO, TIMESTEPS_FILE_PPO
            else:
                true_q_file, q_hat_file, timesteps_file = TRUE_Q_FILE_DDPG, Q_HAT_FILE_DDPG, TIMESTEPS_FILE_DDPG

            true_q_raw = np.load(os.path.join(exp_dir, true_q_file), allow_pickle=True)
            true_q = rowwise_mean(np.asarray(true_q_raw, dtype=np.float64))

            q_hat_raw = np.load(os.path.join(exp_dir, q_hat_file), allow_pickle=True)
            q_hat = rowwise_mean(np.asarray(q_hat_raw, dtype=np.float64))

            steps_raw = np.load(os.path.join(exp_dir, timesteps_file), allow_pickle=True)
            steps = rowwise_mean(np.asarray(steps_raw, dtype=np.int64))

            # Apply *8 scaling for PPO timesteps
            # if algorithm == 'PPO':
                # steps = steps * 8

        except FileNotFoundError as e:
            print(f"Warning: File not found in {exp_dir} - {e}")
            continue

        true_q_curves.append(true_q)
        q_hat_curves.append(q_hat)
        all_timesteps.append(steps)

    if not true_q_curves:
        return None, None, None, None

    # min_len = min(len(c) for c in true_q_curves)
    # true_q_curves = [c[:min_len] for c in true_q_curves]
    # q_hat_curves = [c[:min_len] for c in q_hat_curves]
    # all_timesteps = [c[:min_len] for c in all_timesteps]

    true_q_stack = np.stack(true_q_curves)
    q_hat_stack = np.stack(q_hat_curves)
    timesteps_stack = np.stack(all_timesteps)

    true_q_mean = true_q_stack.mean(axis=0)
    true_q_std = true_q_stack.std(axis=0)
    q_hat_mean = q_hat_stack.mean(axis=0)
    timesteps = timesteps_stack.mean(axis=0)

    return timesteps, true_q_mean, true_q_std, q_hat_mean

# =============================================================================
# --- 3. MAIN PLOTTING FUNCTION ---
# =============================================================================

def generate_plots(arch_dirs):
    """Orchestrates the data loading, processing, and plotting for all experiments."""
    num_archs = len(arch_dirs)
    
    subplot_rows = int(np.ceil(num_archs / SUBPLOT_COLS))
    fig, axes = plt.subplots(
        subplot_rows, SUBPLOT_COLS,
        figsize=(SUBPLOT_COLS * 4.5, subplot_rows * 2.8),
        sharex=True, sharey=True
    )
    axes = axes.flatten() if num_archs > 1 else [axes]

    for i, arch_name in enumerate(arch_dirs):
        ax = axes[i]
        for algo in ALGORITHMS:
            timesteps, true_q_mean, true_q_std, q_hat_mean = process_experiment_data(
                RESULTS_DIR, algo, arch_name, SEEDS
            )

            color = ALGO_COLORS.get(algo, 'gray')
            display_name = ALGO_DISPLAY_NAMES.get(algo, algo)

            # Set RPI to appear on top (higher z-order)
            zorder = 10 if algo == 'DDPG_RPI' else 5

            ax.plot(timesteps, true_q_mean, color=color, linestyle='-', label=f'{display_name} Return', linewidth=3, zorder=zorder)
            ax.fill_between(timesteps, true_q_mean - true_q_std, true_q_mean + true_q_std, color=color, alpha=0.2, zorder=zorder)
            if algo != "PPO_0.999":
                ax.plot(timesteps, q_hat_mean, color=color, linestyle='--', label=f'{display_name} Critic Estimate', linewidth=2, zorder=zorder)
            ax.set_ylim(0, 125)
            ax.set_xlim(0,1e5)

        arch_label = arch_name.replace('arch_', '').split("_")[0]
        ax.set_title(f'{arch_label}')
        ax.grid(True, which='both', linestyle='--', linewidth=1.5)

        # --- MODIFICATION: Set specific x-axis tick positions and labels ---
        ax.set_xticks([0, 0.5e5, 1e5])

    # --- MODIFICATION: Set custom tick labels for all active subplots ---
    for i in range(num_archs):
        axes[i].set_xticklabels(['0', '0.5', '1'])

    for i in range(num_archs, len(axes)):
        fig.delaxes(axes[i])

    # --- MODIFICATION: Refined custom legend ---
    legend_elements = []
    for algo in ALGORITHMS:
        if algo in ALGO_COLORS:
            legend_elements.append(
                Line2D([0], [0], color=ALGO_COLORS[algo], lw=3, label=ALGO_DISPLAY_NAMES[algo])
            )

    ncol = min(len(legend_elements), 4)
    fig.legend(handles=legend_elements, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 1))

    # --- MODIFICATION: Updated axes labels with larger font ---
    fig.text(0.5, 0.0, r'Environment Timesteps ($\times 10^5$)', ha='center', va='center', fontsize=18)
    fig.text(0.06, 0.5, 'Return', ha='center', va='center', rotation='vertical', fontsize=18)

    # plt.tight_layout(rect=[0.08, 0.05, 0.98, 0.95])
    plt.tight_layout(
        rect=[0.08, 0.05, 0.98, 0.92],  # [left, bottom, right, top] margins
        pad=1,      # Padding between figure edge and subplots (default: 1.08)
        h_pad=0.2,    # Height padding between subplots (default: 1.08)
        w_pad=0.2     # Width padding between subplots (default: 1.08)
    )
    plt.savefig(OUTPUT_FILENAME_PDF, bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight', dpi=300)
    print(f"\nPlot successfully generated and saved to '{OUTPUT_FILENAME}'")
    plt.show()


# =============================================================================
# --- 4. TABLE GENERATION FUNCTION ---
# =============================================================================
def generate_table_data(arch_dirs):
    """Calculates aggregate metrics (Final Performance and AUC) and generates a pandas DataFrame."""
    if not arch_dirs:
        print("No architectures to process for the table after exclusions.")
        return

    print("\n--- Generating Table Data ---")
    table_data = []

    for arch_name in arch_dirs:
        row_data = {'Net-Arch': arch_name.replace('arch_', '')}
        for algo in ALGORITHMS:
            final_perf_per_seed = []
            aucs_per_seed = []

            for seed in SEEDS:
                timesteps, true_q_seed, _, _ = process_experiment_data(
                    RESULTS_DIR, algo, arch_name, [seed]
                )

                if timesteps is not None and len(timesteps) > 0 and true_q_seed is not None and len(true_q_seed) > 0:
                    num_final_points = max(1, int(len(true_q_seed) * 0.1))
                    final_perf_per_seed.append(np.mean(true_q_seed[-num_final_points:]))
                    aucs_per_seed.append(np.trapz(true_q_seed, timesteps))

            if final_perf_per_seed:
                final_perf_mean, final_perf_std = np.mean(final_perf_per_seed), np.std(final_perf_per_seed)
                auc_mean, auc_std = np.mean(aucs_per_seed), np.std(aucs_per_seed)

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
    print("\n--- Results Table ---")
    print(df.to_string())

    csv_filename = 'results_summary_pendulum.csv'
    df.to_csv(csv_filename)
    print(f"\nTable saved to {csv_filename}")

    latex_filename = 'results_summary_pendulum.tex'
    df.to_latex(latex_filename, longtable=False)
    print(f"Table saved to {latex_filename}")


# =============================================================================
# --- 5. SCRIPT EXECUTION ---
# =============================================================================
if __name__ == '__main__':
    try:
        arch_discovery_path = os.path.join(RESULTS_DIR, ALGORITHMS[0])
        if not os.path.isdir(arch_discovery_path):
             print(f"FATAL: The directory for the first algorithm '{arch_discovery_path}' was not found.")
             exit()

        all_dirs = os.listdir(arch_discovery_path)
        print(f"all_dirs: {all_dirs}")

        arch_dirs_to_process = sorted(
            [d for d in all_dirs if d.startswith('arch_') and d not in EXCLUDED_ARCHS],
            key=natural_sort_key
        )

    except FileNotFoundError:
        print(f"FATAL: The results directory '{RESULTS_DIR}' was not found.")
        exit()

    if not arch_dirs_to_process:
        print(f"FATAL: No plottable 'arch_' directories found in '{arch_discovery_path}' (after exclusions).")
        exit()

    print(f"Found {len(arch_dirs_to_process)} architectures to process after exclusions.")
    if EXCLUDED_ARCHS:
        print(f"Excluded architectures: {', '.join(EXCLUDED_ARCHS)}")

    generate_plots(arch_dirs_to_process)
    generate_table_data(arch_dirs_to_process)
