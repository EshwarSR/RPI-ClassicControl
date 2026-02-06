# =============================================================================
#
# Environment Change Plotting Script (InvertedPendulum-v5)
#
# Description:
# This script loads reinforcement learning experiment data for DDPG_RPI, DDPG,
# TD3, and PPO on the InvertedPendulum-v5 environment with varying environmental
# parameters. It processes data over multiple seeds, calculates mean and
# standard deviation, and generates publication-quality plots showing
# performance under different gravity, cart mass, and pole mass conditions.
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
BASE_RESULTS_PATH = './results/env-config/InvertedPendulum-v5'
HYPERPARAMETER_CONFIG = 'c_2.0_lambda1_10.0_lambda2_2.0_rmin_1.0'
OUTPUT_DIR = os.path.join('plots', 'ICC', 'InvertedPendulum', 'env-change')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILENAME = os.path.join(
    OUTPUT_DIR, 'env_change_performance_comparison_all_algorithms-FINAL-FINAL.png'
)
OUTPUT_FILENAME_PDF = os.path.join(
    OUTPUT_DIR, 'env_change_performance_comparison_all_algorithms_SMALL.pdf'
)

# --- Algorithm Settings ---
ALGORITHMS = {
    'DDPG_RPI': {
        'name': r'RPI$_{\mathrm{DDPG}}$',
        'base_dir': f'{BASE_RESULTS_PATH}/DDPG_RPI/{HYPERPARAMETER_CONFIG}',
        'color': '#1B9E77',  # Green
        'type': 'actor-critic'  # Uses true_q_vals.npy and q_hat_vals.npy
    },
    'DDPG': {
        'name': 'DDPG',
        'base_dir': f'{BASE_RESULTS_PATH}/DDPG/{HYPERPARAMETER_CONFIG}',
        'color': '#D95F02',  # Orange
        'type': 'actor-critic'
    },
    'TD3': {
        'name': 'TD3',
        'base_dir': f'{BASE_RESULTS_PATH}/TD3/{HYPERPARAMETER_CONFIG}',
        'color': '#095384',  # Purple
        'type': 'actor-critic'
    },
    'PPO': {
        'name': 'PPO',
        'base_dir': './Net-Arch-Exp_PPO/PPO_Runs/results/env-change/InvertedPendulum-v5/PPO',
        'color': '#7570B3',  # Purple-blue
        'type': 'policy-gradient'  # Uses mc_tot_returns.npy and v_s0_estimates.npy
    }
}

# --- Experiment Settings ---

# Environmental parameters to plot (only doubled and halved conditions)
# For DDPG_RPI, DDPG, and TD3
ENV_PARAMS = {
    'gravity': {
        'doubled': 'g_-19.62_cm_10.47197551_pm_5.01859164',
        'halved': 'g_-4.905_cm_10.47197551_pm_5.01859164',
        'title': 'Gravity',
    },
    'mass-cart': {
        'doubled': 'g_-9.81_cm_20.94395102_pm_5.01859164',
        'halved': 'g_-9.81_cm_5.235987755_pm_5.01859164',
        'title': 'Cart Mass',
    },
    'mass-pole': {
        'doubled': 'g_-9.81_cm_10.47197551_pm_10.03718328',
        'halved': 'g_-9.81_cm_10.47197551_pm_2.50929582',
        'title': 'Pole Mass',
    }
}

# PPO uses different directory naming convention
ENV_PARAMS_PPO = {
    'gravity': {
        'doubled': 'arch_default_gravity_-19.62_masscart_10.47197551_masspole_5.01859164',
        'halved': 'arch_default_gravity_-4.905_masscart_10.47197551_masspole_5.01859164',
        'title': 'Gravity',
    },
    'mass-cart': {
        'doubled': 'arch_default_gravity_-9.81_masscart_20.94395102_masspole_5.01859164',
        'halved': 'arch_default_gravity_-9.81_masscart_5.235987755_masspole_5.01859164',
        'title': 'Cart Mass',
    },
    'mass-pole': {
        'doubled': 'arch_default_gravity_-9.81_masscart_10.47197551_masspole_10.03718328',
        'halved': 'arch_default_gravity_-9.81_masscart_10.47197551_masspole_2.50929582',
        'title': 'Pole Mass',
    }
}

SEEDS = list(range(0, 10))  # Using seeds 0-9

# --- Custom display names for plots/legends ---
CONDITION_DISPLAY_NAMES = {
    'doubled': 'Doubled',
    'halved': 'Halved'
}

# --- Data file names ---
TRUE_Q_FILE = 'true_q_vals.npy'
Q_HAT_FILE = 'q_hat_vals.npy'
TIMESTEPS_FILE = 'time_steps.npy'

# --- Plotting Style & Aesthetics ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
    "figure.titlesize": 14
})

# --- Shaded region transparency ---
SHADED_ALPHA = 0.2  # Very transparent to avoid clutter with 6 lines per subplot

# Plot layout - 2 rows x 3 columns (halved/doubled x gravity/cart/pole)
SUBPLOT_ROWS = 2
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


def process_env_condition_data(base_dir, param_type, condition_dir, seeds, algo_type='actor-critic'):
    """Loads and processes data for a single environmental condition and algorithm.

    Args:
        base_dir: Base directory for the specific algorithm
        param_type: Environmental parameter type (e.g., 'gravity', 'mass-cart', 'mass-pole')
        condition_dir: Condition directory name (e.g., 'g_-19.62_cm_10.47197551_pm_5.01859164')
        seeds: List of seeds to process
        algo_type: Type of algorithm ('actor-critic' or 'policy-gradient')

    Returns:
        Tuple of (timesteps, true_q_mean, true_q_std, q_hat_mean)
    """
    true_q_curves, q_hat_curves, all_timesteps = [], [], []

    # Data structure is flat - no param_type subdirectories (gravity/, mass-cart/, mass-pole/)
    exp_base_path = os.path.join(base_dir, condition_dir)

    for seed in seeds:
        # PPO uses 'seed_0', 'seed_1', etc.; actor-critic uses '0', '1', etc.
        seed_dir_name = f'seed_{seed}' if algo_type == 'policy-gradient' else str(seed)
        exp_dir = os.path.join(exp_base_path, seed_dir_name)

        if not os.path.isdir(exp_dir):
            continue

        try:
            if algo_type == 'policy-gradient':
                # PPO: Load mc_tot_returns.npy and v_s0_estimates.npy
                true_q_raw = np.load(os.path.join(exp_dir, 'mc_disc_returns.npy'), allow_pickle=True)
                # PPO data has shape (100, 100, 1), we need to squeeze and take mean over episodes
                true_q_raw = np.asarray(true_q_raw, dtype=np.float64)
                if true_q_raw.ndim == 3:
                    true_q_raw = true_q_raw.squeeze(-1)  # Remove last dimension: (100, 100, 1) -> (100, 100)
                # Take mean over the 100 evaluation episodes for each timestep
                true_q = true_q_raw.mean(axis=1)  # (100, 100) -> (100,)

                q_hat_raw = np.load(os.path.join(exp_dir, 'v_s0_estimates.npy'), allow_pickle=True)
                q_hat_raw = np.asarray(q_hat_raw, dtype=np.float64)
                # Take mean over the 100 evaluation episodes for each timestep
                q_hat = q_hat_raw.mean(axis=1)  # (100, 100) -> (100,)

                steps_raw = np.load(os.path.join(exp_dir, 'time_steps_of_eval.npy'), allow_pickle=True)
                steps_raw = np.asarray(steps_raw, dtype=np.int64)
                # Take the first evaluation's timesteps (they're all the same)
                steps = steps_raw[:, 0]  # (100, 100) -> (100,)
            else:
                # Actor-critic: Load true_q_vals.npy and q_hat_vals.npy
                true_q_raw = np.load(os.path.join(exp_dir, TRUE_Q_FILE), allow_pickle=True)
                true_q = rowwise_mean(np.asarray(true_q_raw, dtype=np.float64))

                q_hat_raw = np.load(os.path.join(exp_dir, Q_HAT_FILE), allow_pickle=True)
                q_hat = rowwise_mean(np.asarray(q_hat_raw, dtype=np.float64))

                steps_raw = np.load(os.path.join(exp_dir, TIMESTEPS_FILE), allow_pickle=True)
                steps = rowwise_mean(np.asarray(steps_raw, dtype=np.int64))

        except FileNotFoundError as e:
            print(f"Warning: File not found in {exp_dir} - {e}")
            continue
        except Exception as e:
            print(f"Warning: Error processing {exp_dir} - {e}")
            continue

        true_q_curves.append(true_q)
        q_hat_curves.append(q_hat)
        all_timesteps.append(steps)

    if not true_q_curves:
        return None, None, None, None

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


def generate_plots():
    """Orchestrates the data loading, processing, and plotting for environment change experiments.
    Creates a 2x3 grid: rows = conditions (halved/doubled), columns = parameters (gravity/cart/pole)
    """
    # Create 2 rows, 3 columns of subplots
    fig, axes = plt.subplots(
        SUBPLOT_ROWS, SUBPLOT_COLS,
        figsize=(SUBPLOT_COLS * 4.5, SUBPLOT_ROWS * 2.8),
        sharex=True, sharey=True
    )

    # ['gravity', 'mass-cart', 'mass-pole']
    param_types = list(ENV_PARAMS.keys())
    # Top row = halved, bottom row = doubled
    conditions = ['halved', 'doubled']
    algorithm_keys = list(ALGORITHMS.keys())  # ['DDPG_RPI', 'DDPG', 'TD3', 'PPO']

    # Plot DDPG_RPI last so it appears on top (z-order)
    if 'DDPG_RPI' in algorithm_keys:
        algorithm_keys.remove('DDPG_RPI')
        algorithm_keys.append('DDPG_RPI')

    # Loop through rows (conditions) and columns (parameters)
    for row_idx, condition in enumerate(conditions):
        for col_idx, param_type in enumerate(param_types):
            ax = axes[row_idx, col_idx]

            # Plot each algorithm with different colors
            for algo_key in algorithm_keys:
                algo_config = ALGORITHMS[algo_key]
                algo_name = algo_config['name']
                algo_color = algo_config['color']
                base_dir = algo_config['base_dir']
                algo_type = algo_config['type']

                # Set z-order: DDPG-RPI gets highest z-order to appear on top
                zorder = 10 if algo_key == 'DDPG_RPI' else 5

                # Use appropriate ENV_PARAMS based on algorithm type
                if algo_type == 'policy-gradient':
                    param_config = ENV_PARAMS_PPO[param_type]
                else:
                    param_config = ENV_PARAMS[param_type]

                condition_dir = param_config[condition]

                timesteps, true_q_mean, true_q_std, q_hat_mean = process_env_condition_data(
                    base_dir, param_type, condition_dir, SEEDS, algo_type
                )

                if timesteps is None:
                    print(
                        f"Warning: No data found for {algo_key} - {param_type} - {condition}")
                    continue

                # Plot return (solid line) with very transparent shaded std
                ax.plot(timesteps, true_q_mean, color=algo_color, linestyle='-',
                        linewidth=2.5, label=f'{algo_name} Return', zorder=zorder)
                ax.fill_between(timesteps, true_q_mean - true_q_std, true_q_mean + true_q_std, color=algo_color, alpha=SHADED_ALPHA, zorder=zorder-1)
                
                if algo_key!="PPO":
                # Plot critic estimate (dashed line)
                    ax.plot(timesteps, q_hat_mean, color=algo_color, linestyle='--',
                        linewidth=2, label=f'{algo_name} Critic', zorder=zorder)

            # Set plot properties
            ax.set_ylim(0, 125)
            ax.set_xlim(0, 1e5)
            ax.grid(True, which='both', linestyle='--', linewidth=1.5)

            # Set x-axis ticks
            ax.set_xticks([0, 0.5e5, 1e5])
            ax.set_xticklabels(['0', '0.5', '1'])

            # Add titles only to top row - use the first param_config (they all have same title)
            if row_idx == 0:
                ax.set_title(ENV_PARAMS[param_type]['title'], fontsize=17)

            # Add row labels on the left
            if col_idx == 0:
                ax.text(-0.18, 0.5, CONDITION_DISPLAY_NAMES[condition],
                        transform=ax.transAxes, fontsize=19, rotation=90,
                        ha='center', va='center')

    # Create custom legend for algorithms (not conditions)
    legend_elements = []
    for algo_key in algorithm_keys:
        algo_config = ALGORITHMS[algo_key]
        legend_elements.append(
            Line2D([0], [0], color=algo_config['color'], lw=3,
                   label=algo_config['name'])
        )

    fig.legend(handles=legend_elements, loc='upper center', ncol=len(algorithm_keys),
               bbox_to_anchor=(0.5, 1.0), fontsize=17)

    # Add axis labels
    fig.text(0.5, 0.02, r'Environment Timesteps ($\times 10^5$)',
             ha='center', va='center', fontsize=18)
    fig.text(0.06, 0.5, 'Return', ha='center', va='center',
             rotation='vertical', fontsize=18)

    # plt.tight_layout(rect=[0.05, 0.06, 0.98, 0.96])
    plt.tight_layout(
        rect=[0.08, 0.05, 0.98, 0.92],  # [left, bottom, right, top] margins
        pad=1,      # Padding between figure edge and subplots (default: 1.08)
        h_pad=0.2,    # Height padding between subplots (default: 1.08)
        w_pad=0.2     # Width padding between subplots (default: 1.08)
    )
    plt.savefig(OUTPUT_FILENAME, dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_FILENAME_PDF, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully generated and saved to '{OUTPUT_FILENAME}'")
    plt.show()


# =============================================================================
# --- 4. TABLE GENERATION FUNCTION ---
# =============================================================================
def generate_table_data():
    """Calculates aggregate metrics (Final Performance and AUC) for environment changes across all algorithms."""
    print("\n--- Generating Table Data ---")
    table_data = []

    # Loop through all algorithms, parameters, and conditions
    for algo_key in ALGORITHMS.keys():
        algo_config = ALGORITHMS[algo_key]
        algo_name = algo_config['name']
        base_dir = algo_config['base_dir']
        algo_type = algo_config['type']

        for param_type in ENV_PARAMS.keys():
            # Use appropriate ENV_PARAMS based on algorithm type
            if algo_type == 'policy-gradient':
                param_config = ENV_PARAMS_PPO[param_type]
            else:
                param_config = ENV_PARAMS[param_type]

            for condition in ['doubled', 'halved']:
                row_data = {
                    'Algorithm': algo_name,
                    'Parameter': ENV_PARAMS[param_type]['title'],  # Use consistent title
                    'Condition': CONDITION_DISPLAY_NAMES[condition]
                }

                final_perf_per_seed = []
                aucs_per_seed = []

                condition_dir = param_config[condition]

                for seed in SEEDS:
                    timesteps, true_q_seed, _, _ = process_env_condition_data(
                        base_dir, param_type, condition_dir, [seed], algo_type
                    )

                    if timesteps is not None and len(timesteps) > 0 and true_q_seed is not None and len(true_q_seed) > 0:
                        num_final_points = max(1, int(len(true_q_seed) * 0.1))
                        final_perf_per_seed.append(
                            np.mean(true_q_seed[-num_final_points:]))
                        aucs_per_seed.append(np.trapz(true_q_seed, timesteps))

                if final_perf_per_seed:
                    final_perf_mean, final_perf_std = np.mean(
                        final_perf_per_seed), np.std(final_perf_per_seed)
                    auc_mean, auc_std = np.mean(
                        aucs_per_seed), np.std(aucs_per_seed)

                    row_data['Final Performance'] = f"{final_perf_mean:.2f} ± {final_perf_std:.2f}"
                    row_data['AUC'] = f"{auc_mean:.0f} ± {auc_std:.0f}"
                else:
                    row_data['Final Performance'] = 'N/A'
                    row_data['AUC'] = 'N/A'

                table_data.append(row_data)

    if not table_data:
        print("No data was processed successfully. Cannot generate table.")
        return

    df = pd.DataFrame(table_data)
    print("\n--- Results Table ---")
    print(df.to_string(index=False))

    csv_filename = os.path.join(
        OUTPUT_DIR, 'env_change_results_summary_all_algorithms.csv'
    )
    df.to_csv(csv_filename, index=False)
    print(f"\nTable saved to {csv_filename}")

    latex_filename = os.path.join(
        OUTPUT_DIR, 'env_change_results_summary_all_algorithms.tex'
    )
    df.to_latex(latex_filename, index=False, longtable=False)
    print(f"Table saved to {latex_filename}")


# =============================================================================
# --- 5. SCRIPT EXECUTION ---
# =============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("Environment Change Performance Analysis (Multi-Algorithm)")
    print("=" * 80)

    # Check if base results path exists
    if not os.path.isdir(BASE_RESULTS_PATH):
        print(f"FATAL: The results directory '{BASE_RESULTS_PATH}' was not found.")
        exit()

    # Verify all algorithm directories exist
    missing_dirs = []
    for algo_key, algo_config in ALGORITHMS.items():
        if not os.path.isdir(algo_config['base_dir']):
            missing_dirs.append(f"{algo_key}: {algo_config['base_dir']}")

    if missing_dirs:
        print(f"WARNING: Some algorithm directories are missing:")
        for d in missing_dirs:
            print(f"  - {d}")
        print("Continuing with available algorithms...")

    # Verify parameter directories exist for each algorithm
    missing_param_dirs = []
    for algo_key, algo_config in ALGORITHMS.items():
        if not os.path.isdir(algo_config['base_dir']):
            continue  # Skip if base dir doesn't exist

        for param_type in ENV_PARAMS.keys():
            param_dir = os.path.join(algo_config['base_dir'], param_type)
            if not os.path.isdir(param_dir):
                missing_param_dirs.append(f"{algo_key}/{param_type}")

    if missing_param_dirs:
        print(f"WARNING: Some parameter directories are missing:")
        for d in missing_param_dirs:
            print(f"  - {d}")
        print("Continuing with available parameter directories...")

    print(f"\nBase results path: {BASE_RESULTS_PATH}")
    print(f"Processing {len(ALGORITHMS)} algorithms: {', '.join([a['name'] for a in ALGORITHMS.values()])}")
    print(f"Processing {len(ENV_PARAMS)} environmental parameters: {', '.join(ENV_PARAMS.keys())}")
    print(f"Conditions: {', '.join(CONDITION_DISPLAY_NAMES.values())}")
    print(f"Number of seeds: {len(SEEDS)}")

    # Generate plots and tables
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)
    generate_plots()

    print("\n" + "=" * 80)
    print("Generating summary tables...")
    print("=" * 80)
    generate_table_data()

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
