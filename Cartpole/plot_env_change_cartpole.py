# =============================================================================
#
# Environment Change Plotting Script (CartPole-v1) - Generalized
#
# Description:
# It processes data over multiple seeds, calculates mean and standard deviation,
# and generates publication-quality plots showing performance under different
# gravity, cart mass, and pole mass conditions.
# Loads experiment data for RPI-DQN, DQN, DoubleDQN, and PPO on CartPole-v1
# using the folder layout under ./results/env-config:
#   <BASE>/<Algo>/<condition_dir>/<Algo>/seed_*/{mc_disc_returns.npy, v_s0_estimates.npy, time_steps_of_eval.npy}
# Grouping into Gravity / Cart Mass / Pole Mass is sourced from ENV_PARAMS mapping.
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
BASE_RESULTS_PATH = './results/env-config'
OUTPUT_DIR = os.path.join('plots', 'ICC', 'Cartpole', 'env-change')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILENAME = os.path.join(
    OUTPUT_DIR, 'env_change_performance_comparison_all_algorithms_cartpole-GENERAL.png'
)
OUTPUT_FILENAME_PDF = os.path.join(
    OUTPUT_DIR, 'env_change_performance_comparison_all_algorithms_cartpole-GENERAL.pdf'
)

# --- Algorithm Settings ---
ALGORITHMS = {
    'RPI-DQN': {
        'name': r'RPI$_{\mathrm{DQN}}$',
        'base_dir': f'{BASE_RESULTS_PATH}/RPI',  # outer folder
        'subdir': 'RPI',  # inner folder within each condition
        'color': '#1B9E77'  # Green
    },
    'DQN': {
        'name': 'DQN',
        'base_dir': f'{BASE_RESULTS_PATH}/DQN',
        'subdir': 'DQN',
        'color': '#D95F02'  # Orange
    },
    'PPO': {
        'name': 'PPO',
        'base_dir': f'{BASE_RESULTS_PATH}/PPO',
        'subdir': 'PPO',
        'color': '#7570B3'  # Purple
    },
    'DoubleDQN': {
        'name': 'DoubleDQN',
        'base_dir': f'{BASE_RESULTS_PATH}/DoubleDQN',
        'subdir': 'DoubleDQN',
        'color': '#095384'
    }
}

# --- Experiment Settings ---
# Environmental parameters to plot (only doubled and halved conditions)
ENV_PARAMS = {
    'gravity': {
        'doubled': 'g_19.6_mc_1.0_mp_0.1_l_0.5',
        'halved': 'g_4.9_mc_1.0_mp_0.1_l_0.5',
        'title': 'Gravity',
    },
    'mass-cart': {
        'doubled': 'g_9.8_mc_2.0_mp_0.1_l_0.5',
        'halved': 'g_9.8_mc_0.5_mp_0.1_l_0.5',
        'title': 'Cart Mass',
    },
    'mass-pole': {
        'doubled': 'g_9.8_mc_1.0_mp_0.2_l_0.5',
        'halved': 'g_9.8_mc_1.0_mp_0.05_l_0.5',
        'title': 'Pole Mass',
    }
}

SEEDS = list(range(0, 10))  # Using seeds 0-9

# --- Custom display names for plots/legends ---
CONDITION_DISPLAY_NAMES = {
    'doubled': 'Doubled',
    'halved': 'Halved'
}

# --- Data file names (CartPole-specific) ---
TRUE_Q_FILE = 'mc_disc_returns.npy'  # Total returns
Q_HAT_FILE = 'v_s0_estimates.npy'   # Value estimates
TIMESTEPS_FILE = 'time_steps_of_eval.npy'

# --- Plotting Style & Aesthetics ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 15,
    'axes.labelsize': 18,
    'axes.titlesize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
    'figure.titlesize': 14,
})

# --- Shaded region transparency ---
SHADED_ALPHA = 0.2  # Very transparent to avoid clutter with 6 lines per subplot

# Plot layout - 2 rows x 3 columns (halved/doubled x gravity/cart/pole)
SUBPLOT_ROWS = 2
SUBPLOT_COLS = 3

SENSITIVITY = 10

# =============================================================================
# --- 2. HELPER FUNCTIONS (Data loading and processing) ---
# =============================================================================

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def rowwise_mean(arr):
    if arr.ndim > 1:
        return arr.reshape(arr.shape[0], -1).mean(axis=1)
    return arr


def process_env_condition_data(base_dir, algo_key, algo_subdir, condition_dir, seeds):
    """Loads and processes data for a single environmental condition and algorithm.

    Folder layout assumed:
        base_dir/<condition_dir>/<algo_subdir>/seed_*
    """
    true_q_curves, q_hat_curves, all_timesteps = [], [], []

    exp_base_path = os.path.join(base_dir, condition_dir, algo_subdir)

    for seed in seeds:
        exp_dir = os.path.join(exp_base_path, f'seed_{seed}')
        if not os.path.isdir(exp_dir):
            continue
        try:
            true_q_raw = np.load(os.path.join(exp_dir, TRUE_Q_FILE), allow_pickle=True)
            true_q = rowwise_mean(np.asarray(true_q_raw, dtype=np.float64))

            q_hat_raw = np.load(os.path.join(exp_dir, Q_HAT_FILE), allow_pickle=True)
            q_hat = rowwise_mean(np.asarray(q_hat_raw, dtype=np.float64))

            steps_raw = np.load(os.path.join(exp_dir, TIMESTEPS_FILE), allow_pickle=True)
            steps = rowwise_mean(np.asarray(steps_raw, dtype=np.int64))
        except FileNotFoundError as e:
            print(f"Warning: File not found in {exp_dir} - {e}")
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

    # PPO uses different timestep convention - multiply by 8
    if algo_key == 'PPO':
        timesteps = timesteps * 8

    return timesteps, true_q_mean, true_q_std, q_hat_mean


# =============================================================================
# --- 3. MAIN PLOTTING FUNCTION ---
# =============================================================================

def generate_plots():
    """Creates a 2x3 grid: rows = conditions (halved/doubled), columns = parameters (gravity/cart/pole)."""
    fig, axes = plt.subplots(
        SUBPLOT_ROWS, SUBPLOT_COLS,
        figsize=(SUBPLOT_COLS * 4.5, SUBPLOT_ROWS * 2.8),
        sharex=True, sharey=True,
    )

    param_types = list(ENV_PARAMS.keys())  # ['gravity', 'mass-cart', 'mass-pole']
    conditions = ['halved', 'doubled']
    algorithm_keys = list(ALGORITHMS.keys())  # ['RPI-DQN', 'DQN', 'DoubleDQN', 'PPO']

    for row_idx, condition in enumerate(conditions):
        for col_idx, param_type in enumerate(param_types):
            ax = axes[row_idx, col_idx]
            param_config = ENV_PARAMS[param_type]
            condition_dir = param_config[condition]

            for algo_key in algorithm_keys:
                algo_config = ALGORITHMS[algo_key]
                algo_name = algo_config['name']
                algo_color = algo_config['color']
                base_dir = algo_config['base_dir']
                algo_subdir = algo_config['subdir']

                timesteps, true_q_mean, true_q_std, q_hat_mean = process_env_condition_data(
                    base_dir, algo_key, algo_subdir, condition_dir, SEEDS
                )

                if timesteps is None:
                    print(f"Warning: No data found for {algo_key} - {param_type} - {condition}")
                    continue

                zorder_value = 10 if algo_key == 'RPI-DQN' else 5

                ax.plot(timesteps[::SENSITIVITY], true_q_mean[::SENSITIVITY], color=algo_color, linestyle='-',
                        linewidth=2.5, label=f'{algo_name} Return', zorder=zorder_value)
                ax.fill_between(
                    timesteps[::SENSITIVITY],
                    true_q_mean[::SENSITIVITY] - true_q_std[::SENSITIVITY],
                    true_q_mean[::SENSITIVITY] + true_q_std[::SENSITIVITY],
                    color=algo_color, alpha=SHADED_ALPHA, zorder=zorder_value - 1,
                )
                ax.plot(
                    timesteps[::SENSITIVITY], q_hat_mean[::SENSITIVITY], color=algo_color, linestyle='--',
                    linewidth=2, label=f'{algo_name} Critic', zorder=zorder_value,
                )

            ax.set_ylim(0, 125)
            ax.set_xlim(0, 1e5)
            ax.grid(True, which='both', linestyle='--', linewidth=1.5)

            ax.set_xticks([0, 0.5e5, 1e5])
            ax.set_xticklabels(['0', '0.5', '1'])

            if row_idx == 0:
                ax.set_title(param_config['title'], fontsize=18)

            if col_idx == 0:
                ax.text(
                    -0.18, 0.5, CONDITION_DISPLAY_NAMES[condition],
                    transform=ax.transAxes, fontsize=17, rotation=90,
                    ha='center', va='center',
                )

    legend_elements = []
    for algo_key in algorithm_keys:
        algo_config = ALGORITHMS[algo_key]
        legend_elements.append(
            Line2D([0], [0], color=algo_config['color'], lw=3, label=algo_config['name'])
        )

    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=17)

    fig.text(0.5, 0.02, r'Environment Timesteps ($\times 10^5$)', ha='center', va='center', fontsize=19)
    fig.text(0.06, 0.5, 'Return', ha='center', va='center', rotation='vertical', fontsize=19)

    plt.tight_layout(
        rect=[0.08, 0.05, 0.98, 0.92],
        pad=1,
        h_pad=0.2,
        w_pad=0.2,
    )
    plt.savefig(OUTPUT_FILENAME, dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_FILENAME_PDF, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully generated and saved to '{OUTPUT_FILENAME}'")
    plt.show()


# =============================================================================
# --- 4. TABLE GENERATION FUNCTION ---
# =============================================================================

def generate_table_data():
    """Calculates aggregate metrics (Final Performance and AUC) for env changes across all algorithms."""
    print("\n--- Generating Table Data ---")
    table_data = []

    for algo_key in ALGORITHMS.keys():
        algo_config = ALGORITHMS[algo_key]
        algo_name = algo_config['name']
        base_dir = algo_config['base_dir']
        algo_subdir = algo_config['subdir']

        for param_type in ENV_PARAMS.keys():
            param_config = ENV_PARAMS[param_type]
            for condition in ['doubled', 'halved']:
                row_data = {
                    'Algorithm': algo_name,
                    'Parameter': param_config['title'],
                    'Condition': CONDITION_DISPLAY_NAMES[condition],
                }

                final_perf_per_seed = []
                aucs_per_seed = []

                condition_dir = param_config[condition]

                for seed in SEEDS:
                    timesteps, true_q_seed, _, _ = process_env_condition_data(
                        base_dir, algo_key, algo_subdir, condition_dir, [seed]
                    )
                    if timesteps is not None and len(timesteps) > 0 and true_q_seed is not None and len(true_q_seed) > 0:
                        num_final_points = max(1, int(len(true_q_seed) * 0.1))
                        final_perf_per_seed.append(np.mean(true_q_seed[-num_final_points:]))
                        aucs_per_seed.append(np.trapz(true_q_seed, timesteps))

                if final_perf_per_seed:
                    final_perf_mean, final_perf_std = np.mean(final_perf_per_seed), np.std(final_perf_per_seed)
                    auc_mean, auc_std = np.mean(aucs_per_seed), np.std(aucs_per_seed)
                    row_data['Final Performance'] = f"{final_perf_mean:.2f} ± {final_perf_std:.2f}"
                    row_data['AUC'] = f"{auc_mean:.0f} ± {auc_std:.0f}"
                else:
                    row_data['Final Performance'] = 'N/A'
                    row_data['AUC'] = 'N/A'

                table_data.append(row_data)

    if not table_data:
        print('No data was processed successfully. Cannot generate table.')
        return

    df = pd.DataFrame(table_data)
    print('\n--- Results Table ---')
    print(df.to_string(index=False))

    csv_filename = os.path.join(
        OUTPUT_DIR, 'env_change_results_summary_all_algorithms_cartpole-GENERAL.csv'
    )
    df.to_csv(csv_filename, index=False)
    print(f"\nTable saved to {csv_filename}")

    latex_filename = os.path.join(
        OUTPUT_DIR, 'env_change_results_summary_all_algorithms_cartpole-GENERAL.tex'
    )
    df.to_latex(latex_filename, index=False, longtable=False)
    print(f"Table saved to {latex_filename}")


# =============================================================================
# --- 5. SCRIPT EXECUTION ---
# =============================================================================
if __name__ == '__main__':
    print('=' * 80)
    print('Environment Change Performance Analysis - CartPole (Multi-Algorithm) - GENERAL')
    print('=' * 80)

    # Check if base results path exists
    if not os.path.isdir(BASE_RESULTS_PATH):
        print(f"FATAL: The results directory '{BASE_RESULTS_PATH}' was not found.")
        exit()

    # Verify all algorithm base directories exist
    missing_dirs = []
    for algo_key, algo_config in ALGORITHMS.items():
        if not os.path.isdir(algo_config['base_dir']):
            missing_dirs.append(algo_config['base_dir'])

    if missing_dirs:
        print('FATAL: Missing algorithm directories:')
        for d in missing_dirs:
            print(f'  - {d}')
        exit()

    # Verify required condition directories exist for each algorithm
    # Collect unique condition dirs from ENV_PARAMS
    required_condition_dirs = set()
    for param in ENV_PARAMS.values():
        required_condition_dirs.add(param['doubled'])
        required_condition_dirs.add(param['halved'])

    missing_condition_dirs = []
    for algo_key, algo_config in ALGORITHMS.items():
        base_dir = algo_config['base_dir']
        algo_subdir = algo_config['subdir']
        for cond_dir in required_condition_dirs:
            path = os.path.join(base_dir, cond_dir, algo_subdir)
            if not os.path.isdir(path):
                missing_condition_dirs.append(path)

    if missing_condition_dirs:
        print('FATAL: Missing condition directories (expected layout base/cond/algo/):')
        for d in missing_condition_dirs:
            print(f'  - {d}')
        exit()

    print(f"\nBase results path: {BASE_RESULTS_PATH}")
    print(f"Processing {len(ALGORITHMS)} algorithms: {', '.join([a['name'] for a in ALGORITHMS.values()])}")
    print(f"Processing {len(ENV_PARAMS)} environmental parameters: {', '.join(ENV_PARAMS.keys())}")
    print(f"Conditions: {', '.join(CONDITION_DISPLAY_NAMES.values())}")
    print(f"Number of seeds: {len(SEEDS)}")

    print('\n' + '=' * 80)
    print('Generating plots...')
    print('=' * 80)
    generate_plots()

    print('\n' + '=' * 80)
    print('Generating summary tables...')
    print('=' * 80)
    generate_table_data()

    print('\n' + '=' * 80)
    print('Analysis complete!')
    print('=' * 80)
