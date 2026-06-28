"""Experiment harness: run parameter sweeps and collect summary metrics."""

import itertools
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from src.simulation import Simulation
from src.agents import (
    FAILURE_STREAK, ALPHA_LEARN_RATE, REWARD_SCALE, NEST_PULL, CROWD_AVOID,
)
from src.analysis import (
    compute_collective_metrics,
    compute_role_diversity_timeseries,
    compute_agent_metrics,
    compute_exploitation_index,
    per_run_role_fractions,
)

RESULTS_DIR = Path(__file__).parent.parent / 'results'


def run_experiment(
    patch_distribution='gaussian',
    pheromones_enabled=True,
    num_agents=100,
    steps=300,
    seed=42,
    depletion_rate=0.07,
    crowding_sensitivity=1.0,
    sensing_radius=6.0,
    pheromone_sensing_radius=None,
    exploiter_fraction=0.5,
    n_items=150,
    pheromone_weight=1.0,
    energy_return_threshold=40.0,
    failure_streak=FAILURE_STREAK,
    alpha_learn_rate=ALPHA_LEARN_RATE,
    reward_scale=REWARD_SCALE,
    nest_pull=NEST_PULL,
    crowd_avoid=CROWD_AVOID,
):
    """Run one simulation configuration and return (summary_row, dataframes)."""
    sim = Simulation(
        num_agents=num_agents,
        patch_distribution=patch_distribution,
        pheromones_enabled=pheromones_enabled,
        seed=seed,
        depletion_rate=depletion_rate,
        crowding_sensitivity=crowding_sensitivity,
        sensing_radius=sensing_radius,
        pheromone_sensing_radius=pheromone_sensing_radius,
        exploiter_fraction=exploiter_fraction,
        n_items=n_items,
        pheromone_weight=pheromone_weight,
        energy_return_threshold=energy_return_threshold,
        failure_streak=failure_streak,
        alpha_learn_rate=alpha_learn_rate,
        reward_scale=reward_scale,
        nest_pull=nest_pull,
        crowd_avoid=crowd_avoid,
    )
    agent_df, env_df, swarm_df, switch_df = sim.run(steps)

    metrics      = compute_collective_metrics(agent_df, swarm_df)
    diversity_ts = compute_role_diversity_timeseries(agent_df)
    final_roles  = agent_df.groupby('agent_id')['role'].last().value_counts(normalize=True)
    last_step    = agent_df[agent_df['step'] == agent_df['step'].max()]

    row = {
        'patch_distribution':  patch_distribution,
        'pheromones_enabled':  pheromones_enabled,
        'num_agents':          num_agents,
        'steps':               steps,
        'seed':                seed,
        'depletion_rate':      depletion_rate,
        'crowding_sensitivity': crowding_sensitivity,
        'sensing_radius':      sensing_radius,
        'pheromone_sensing_radius': (
            pheromone_sensing_radius if pheromone_sensing_radius is not None
            else sensing_radius
        ),
        'exploiter_fraction':  exploiter_fraction,
        'n_items':             n_items,
        'pheromone_weight':         pheromone_weight,
        'energy_return_threshold':  energy_return_threshold,
        'failure_streak':           failure_streak,
        'alpha_learn_rate':         alpha_learn_rate,
        'reward_scale':             reward_scale,
        'nest_pull':                nest_pull,
        'crowd_avoid':              crowd_avoid,
        **metrics,
        'n_role_switch_events': len(switch_df),
        'shannon_final': (
            diversity_ts['shannon_diversity'].iloc[-1] if len(diversity_ts) else 0.0
        ),
        'mean_alpha_final': float(last_step['alpha'].mean()),
        'mean_beta_final':  float(last_step['beta'].mean()),
        'crisis_steps':     int(swarm_df['crisis_active'].sum()),
    }
    for role, frac in final_roles.items():       # last-frame mix (back-compat)
        row[f'final_{role}_frac'] = frac

    # Window-averaged role-mix fractions (supersede the last-frame final_<role>_frac).
    row.update(per_run_role_fractions(agent_df, swarm_df))

    # Role-resolved agent-level metrics explaining why roles differ.
    agent_metrics = compute_agent_metrics(agent_df)
    if not agent_metrics.empty:
        for role in ('exploiter', 'explorer', 'switcher'):
            sub = agent_metrics[agent_metrics['alpha_role_window'] == role]
            row[f'dist_nest_{role}']     = float(sub['dist_nest_mean'].mean()) if len(sub) else np.nan
            row[f'local_resource_{role}'] = float(sub['local_resource_mean'].mean()) if len(sub) else np.nan
            row[f'yield_per_dist_{role}'] = float(sub['yield_per_distance'].mean()) if len(sub) else np.nan
        if agent_metrics['alpha_mean_window'].nunique() > 1:
            row['corr_distnest_alpha'] = float(
                agent_metrics['dist_nest_mean'].corr(agent_metrics['alpha_mean_window'])
            )
            # Headline per-run statistic: corr(FEI, alpha), used by the OAT sweep.
            fei_df, _ = compute_exploitation_index(agent_metrics)
            row['corr_fei_alpha'] = float(
                fei_df['exploitation_index'].corr(fei_df['alpha_mean_window'])
            )
        else:
            row['corr_distnest_alpha'] = np.nan
            row['corr_fei_alpha'] = np.nan

    return row, agent_df, env_df, swarm_df, switch_df


def run_sweep(sweep_params, fixed_params=None, n_seeds=3, base_seed=42,
              save_csv=True):
    """Sweep the Cartesian product of sweep_params (each combo over n_seeds),
    holding fixed_params constant; returns one row per run."""
    if not sweep_params:
        raise ValueError("sweep_params is empty — list at least one param to sweep.")

    fixed_params = dict(fixed_params or {})
    names   = list(sweep_params.keys())
    combos  = list(itertools.product(*(list(sweep_params[n]) for n in names)))
    total   = len(combos) * n_seeds
    done    = 0
    results = []

    for combo in combos:
        combo_kwargs = dict(zip(names, combo))
        run_kwargs   = {**fixed_params, **combo_kwargs}   # swept values override fixed
        run_kwargs.pop('seed', None)                      # seed is set per-iteration
        for seed_offset in range(n_seeds):
            seed = base_seed + seed_offset
            done += 1
            label = ", ".join(f"{k}={v}" for k, v in combo_kwargs.items())
            print(f"[{done}/{total}] {label}, seed={seed}")
            row, *_ = run_experiment(seed=seed, **run_kwargs)
            results.append(row)

    df = pd.DataFrame(results)
    if save_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = RESULTS_DIR / f'sweep_{ts}.csv'
        df.to_csv(path, index=False)
        print(f"Saved to {path}")
    return df
