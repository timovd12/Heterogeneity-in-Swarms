import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROLE_PALETTE = {
    'exploiter': '#d62728',
    'explorer':  '#1f77b4',
    'switcher':  '#ff7f0e',
}

# alpha cut-points for the threshold role (mirrors Agent._infer_role).
ROLE_THRESHOLDS = (0.4, 0.6)


def role_from_alpha(alpha, thresholds=ROLE_THRESHOLDS):
    """Map an α value to a role label: explorer < lo <= switcher <= hi < exploiter."""
    lo, hi = thresholds
    if alpha > hi:
        return 'exploiter'
    if alpha < lo:
        return 'explorer'
    return 'switcher'


# ── Diversity metrics ─────────────────────────────────────────────────────────

def shannon_diversity(proportions):
    p = np.asarray(proportions, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def compute_role_diversity_timeseries(agent_df):
    rows = []
    for step, grp in agent_df.groupby('step'):
        fracs = grp['role'].value_counts(normalize=True)
        rows.append({
            'step': step,
            'shannon_diversity': shannon_diversity(fracs.values),
            'n_roles_active': int((fracs > 0).sum()),
        })
    return pd.DataFrame(rows)


def compute_collective_metrics(agent_df, swarm_df):
    total_collected = swarm_df['total_resource_collected'].sum()
    total_distance  = agent_df.groupby('agent_id')['distance_traveled'].last().sum()
    efficiency      = total_collected / max(total_distance, 1e-6)
    diversity_ts    = compute_role_diversity_timeseries(agent_df)
    mean_shannon    = diversity_ts['shannon_diversity'].mean()
    coverage = (
        swarm_df['coverage_fraction'].iloc[-1]
        if 'coverage_fraction' in swarm_df else np.nan
    )
    return {
        'total_collected':      total_collected,
        'total_distance':       total_distance,
        'efficiency':           efficiency,
        'mean_shannon_diversity': mean_shannon,
        'final_coverage':       coverage,
    }


# ── Agent-level behavioural metrics ───────────────────────────────────────────

AGENT_METRIC_COLS = [
    # spatial anchoring
    'dist_nest_mean', 'dist_nest_median', 'dist_nest_p90', 'frac_time_near_nest',
    'gyration_radius',
    # experienced richness
    'local_resource_mean', 'harvest_success_rate', 'time_since_success_mean',
    # trips & efficiency
    'n_deliveries', 'deliveries_per_100steps', 'total_delivered',
    'window_distance', 'yield_per_distance',
    # role dynamics over the window
    'alpha_std_window', 'frac_window_exploiter', 'frac_window_explorer',
    'frac_window_switcher',
]

# Window-representative identity columns (added to every agent row).
AGENT_IDENTITY_COLS = [
    'final_role', 'modal_role_window', 'alpha_role_window',
    'final_alpha', 'final_beta', 'alpha_mean_window', 'beta_mean_window',
]


def compute_agent_metrics(agent_df, nest_x=50.0, nest_y=50.0,
                          window_frac=0.5, near_radius=15.0):
    """Per-agent behavioural metrics over the steady-state window (last
    window_frac of the run); one row per agent_id."""
    if agent_df.empty:
        return pd.DataFrame(columns=(['agent_id'] + AGENT_METRIC_COLS + AGENT_IDENTITY_COLS))

    step_min, step_max = agent_df['step'].min(), agent_df['step'].max()
    cutoff = step_min + window_frac * (step_max - step_min)
    win = agent_df[agent_df['step'] >= cutoff]

    rows = []
    for aid, g in win.groupby('agent_id'):
        g = g.sort_values('step')
        xs, ys = g['x'].values, g['y'].values
        dist = np.hypot(xs - nest_x, ys - nest_y)
        # Gyration around the agent's own mean position (home-range tightness).
        gyration = float(np.sqrt(((xs - xs.mean())**2 + (ys - ys.mean())**2).mean()))
        window_distance = float(
            g['distance_traveled'].iloc[-1] - g['distance_traveled'].iloc[0]
        )
        total_delivered = float(g['deposited_resources'].sum())
        n_steps = max(len(g), 1)
        n_deliveries = float(g['deposits_this_step'].sum())
        # window-averaged identity instead of the last frame
        alpha_w = g['alpha'].values
        beta_w = g['beta'].values
        roles_w = g['role'].values
        alpha_mean = float(alpha_w.mean())
        last = g.iloc[-1]
        modal = g['role'].mode()
        rows.append({
            'agent_id': aid,
            'dist_nest_mean':        float(dist.mean()),
            'dist_nest_median':      float(np.median(dist)),
            'dist_nest_p90':         float(np.quantile(dist, 0.9)),
            'frac_time_near_nest':   float((dist < near_radius).mean()),
            'gyration_radius':       gyration,
            'local_resource_mean':   float(g['local_resource_level'].mean()),
            'harvest_success_rate':  float((g['harvested_resources'] > 0).mean()),
            'time_since_success_mean': float(g['time_since_last_success'].mean()),
            'n_deliveries':          n_deliveries,
            'deliveries_per_100steps': n_deliveries / n_steps * 100.0,
            'total_delivered':       total_delivered,
            'window_distance':       window_distance,
            'yield_per_distance':    total_delivered / max(window_distance, 1e-6),
            'alpha_std_window':      float(alpha_w.std()),
            'frac_window_exploiter': float((roles_w == 'exploiter').mean()),
            'frac_window_explorer':  float((roles_w == 'explorer').mean()),
            'frac_window_switcher':  float((roles_w == 'switcher').mean()),
            'final_role':            last['role'],
            'modal_role_window':     modal.iloc[0] if len(modal) else last['role'],
            'alpha_role_window':     role_from_alpha(alpha_mean),
            'final_alpha':           float(last['alpha']),
            'final_beta':            float(last['beta']),
            'alpha_mean_window':     alpha_mean,
            'beta_mean_window':      float(beta_w.mean()),
        })
    return pd.DataFrame(rows)


# Metrics attributed to the agent's per-step current role.
ROLE_CONDITIONED_COLS = ['dist_nest_mean', 'gyration_radius',
                         'local_resource_mean', 'yield_per_distance']


def compute_role_conditioned_metrics(agent_df, nest_x=50.0, nest_y=50.0,
                                     window_frac=0.5, min_steps=5):
    """Behavioural metrics split by the per-step role each agent held over the
    window; one row per (agent_id, current_role). Cells with < min_steps dropped."""
    if agent_df.empty:
        return pd.DataFrame(columns=['agent_id', 'current_role'] + ROLE_CONDITIONED_COLS)

    step_min, step_max = agent_df['step'].min(), agent_df['step'].max()
    cutoff = step_min + window_frac * (step_max - step_min)
    win = agent_df[agent_df['step'] >= cutoff]

    rows = []
    for aid, g in win.groupby('agent_id'):
        g = g.sort_values('step')
        xs, ys = g['x'].values, g['y'].values
        dnest = np.hypot(xs - nest_x, ys - nest_y)
        # per-step distance increment, so distance in each role can be summed
        dstep = np.clip(np.diff(g['distance_traveled'].values, prepend=
                                g['distance_traveled'].values[0]), 0.0, None)
        dep = g['deposited_resources'].values
        loc = g['local_resource_level'].values
        roles = g['role'].values
        for role in ('exploiter', 'switcher', 'explorer'):
            sel = roles == role
            if int(sel.sum()) < min_steps:
                continue
            rx, ry = xs[sel], ys[sel]
            gyr = float(np.sqrt(((rx - rx.mean())**2 + (ry - ry.mean())**2).mean()))
            dist_in_role = float(dstep[sel].sum())
            rows.append({
                'agent_id':            aid,
                'current_role':        role,
                'dist_nest_mean':      float(dnest[sel].mean()),
                'gyration_radius':     gyr,
                'local_resource_mean': float(loc[sel].mean()),
                'yield_per_distance':  float(dep[sel].sum()) / max(dist_in_role, 1e-6),
            })
    return pd.DataFrame(rows)


# ── Agent-level figures ───────────────────────────────────────────────────────

_METRIC_LABELS = {
    'dist_nest_mean':       'Mean distance to nest',
    'dist_nest_p90':        '90th-pct distance (foray range)',
    'frac_time_near_nest':  'Fraction of time near nest',
    'gyration_radius':      'Home-range radius (gyration)',
    'local_resource_mean':  'Local resource experienced',
    'harvest_success_rate': 'Harvest success rate',
    'time_since_success_mean': 'Mean steps since success',
    'deliveries_per_100steps': 'Deliveries per 100 steps',
    'yield_per_distance':   'Yield per unit distance',
    'dist_nest_median':     'Median distance to nest',
    'final_alpha':          'Final α (exploitation)',
    'final_beta':           'Final β (exploration)',
    'alpha_mean_window':    'Mean α over window (exploitation)',
    'alpha_std_window':     'α variability over window (switching)',
    'frac_window_switcher': 'Fraction of window as switcher',
    'frac_window_exploiter': 'Fraction of window as exploiter',
    'frac_window_explorer': 'Fraction of window as explorer',
    'pop_dist_nest_mean':   'Population mean distance to nest',
    'exploitation_index':     'Foraging-Exploitation Index (FEI)',
    'pop_exploitation_index': 'Population mean FEI',
    'current_role':           'Current role',
}


def plot_role_occupancy_heatmap(agent_df, window_frac=0.5, bins=40,
                                width=100.0, height=100.0,
                                nest_x=50.0, nest_y=50.0):
    """2D occupancy histogram of agent positions over the window, one panel per role."""
    step_min, step_max = agent_df['step'].min(), agent_df['step'].max()
    cutoff = step_min + window_frac * (step_max - step_min)
    win = agent_df[agent_df['step'] >= cutoff]
    roles = ['exploiter', 'switcher', 'explorer']
    fig, axes = plt.subplots(1, len(roles), figsize=(5 * len(roles), 5))
    rng = [[0, width], [0, height]]
    for ax, role in zip(axes, roles):
        g = win[win['role'] == role]
        if len(g):
            ax.hist2d(g['x'], g['y'], bins=bins, range=rng, cmap='viridis')
        ax.plot(nest_x, nest_y, marker='*', color='white', markersize=16,
                markeredgecolor='black', label='Nest')
        ax.set_title(f"{role.capitalize()} (n_obs={len(g)})", fontsize=11)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
    fig.suptitle("Where each role spends its time", fontsize=14)
    plt.tight_layout()
    return fig


# ── Pooled / many-run behavioural figures ─────────────────────────────────────


def swarm_ci(swarm_dfs, col, transform=None):
    """Align per-seed swarm_dfs on `step`; return (steps, mean, std) across seeds.
    `transform`: optional fn(sorted_df) -> series instead of reading `col`."""
    series = []
    for sdf in swarm_dfs:
        s = sdf.sort_values('step')
        vals = transform(s) if transform is not None else s[col].values
        series.append(pd.Series(np.asarray(vals, dtype=float), index=s['step'].values))
    mat = pd.concat(series, axis=1)
    steps = mat.index.values
    mean = mat.mean(axis=1).values
    std = (mat.std(axis=1, ddof=1).values if mat.shape[1] > 1
           else np.zeros(len(steps)))
    return steps, mean, std


def plot_role_fractions_over_time_ci(env_swarm_by):
    """One panel per environment; each: one line+band per role (role palette)."""
    roles = ['exploiter', 'switcher', 'explorer']
    names = list(env_swarm_by.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.3 * len(names), 5),
                             sharey=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        dfs = env_swarm_by[name]
        for role in roles:
            steps, mean, std = swarm_ci(dfs, f"{role}_fraction")
            c = ROLE_PALETTE.get(role, 'gray')
            ax.plot(steps, mean, color=c, linewidth=2, label=role.capitalize())
            ax.fill_between(steps, mean - std, mean + std, color=c, alpha=0.2)
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Fraction of agents")
    axes[0].legend(fontsize=8)
    fig.suptitle("Role fractions over time (mean ±std across seeds)", fontsize=14)
    plt.tight_layout()
    return fig


def plot_popmean_vs_exploiter_fraction(env_summary, x='pop_dist_nest_mean',
                                       y='timeavg_window_exploiter_frac', by='environment'):
    """One point per run (env × seed): population-mean x vs exploiter fraction y,
    with the overall Pearson r and linear fit annotated."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('tab10')
    for i, n in enumerate(list(dict.fromkeys(env_summary[by]))):
        g = env_summary[env_summary[by] == n]
        ax.scatter(g[x], g[y], color=cmap(i % 10), alpha=0.75, s=32, label=n)
    xv = np.asarray(env_summary[x].values, dtype=float)
    yv = np.asarray(env_summary[y].values, dtype=float)
    m = np.isfinite(xv) & np.isfinite(yv)
    if m.sum() > 1 and xv[m].min() != xv[m].max():
        r = np.corrcoef(xv[m], yv[m])[0, 1]
        b1, b0 = np.polyfit(xv[m], yv[m], 1)
        xs = np.linspace(xv[m].min(), xv[m].max(), 50)
        ax.plot(xs, b0 + b1 * xs, 'k--', alpha=0.7, label=f"fit (r={r:+.2f})")
    xlab = _METRIC_LABELS.get(x, x)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Exploiter fraction")
    ax.set_title(f"Population {xlab.lower()} vs exploiter fraction (one point per run)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ── Statistical helpers (effect sizes, CIs, assumptions, reporting) ────────────
# Tests use the run (seed) as the independent unit, not individual agents.

def bootstrap_ci(values, statistic=np.mean, n_resamples=10000, ci=0.95, seed=0):
    """Percentile bootstrap CI for a 1-D statistic; returns (point, ci_low, ci_high)."""
    from scipy import stats as st
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    point = float(statistic(v))
    if len(v) < 2 or np.std(v) == 0:
        return point, np.nan, np.nan
    res = st.bootstrap((v,), statistic, n_resamples=n_resamples,
                       confidence_level=ci, method='percentile',
                       random_state=np.random.default_rng(seed))
    return point, float(res.confidence_interval.low), float(res.confidence_interval.high)


def cohens_d(a, b):
    """Cohen's d for two independent samples using the pooled standard deviation."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def _eta_squared_anova(F, df_between, df_within):
    """eta^2 from a one-way ANOVA F: SS_between / SS_total."""
    denom = df_between * F + df_within
    return float(df_between * F / denom) if denom > 0 else np.nan


def _epsilon_squared_kruskal(H, n):
    """epsilon^2 effect size for Kruskal-Wallis (Tomczak & Tomczak): H / (n - 1)."""
    return float(H / (n - 1)) if n > 1 else np.nan


def adjust_pvalues(pvals, method='fdr_bh'):
    """Multiple-comparison correction, method='fdr_bh' or 'holm'; adjusted
    p-values in the original order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ps = p[order]
    adj = np.empty(m, dtype=float)
    if method == 'holm':
        running = 0.0
        for i in range(m):
            running = max(running, (m - i) * ps[i])
            adj[i] = min(running, 1.0)
    else:  # Benjamini-Hochberg FDR
        running = 1.0
        for i in range(m - 1, -1, -1):
            running = min(running, m * ps[i] / (i + 1))
            adj[i] = min(running, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out


def check_test_assumptions(env_summary,
                           metrics=('timeavg_window_exploiter_frac',
                                    'timeavg_window_explorer_frac',
                                    'mean_shannon_diversity', 'pop_dist_nest_mean'),
                           by='environment'):
    """Per-metric Shapiro-Wilk (normality) and Levene (homogeneity) diagnostics;
    parametric_ok is True only when both hold at alpha=.05."""
    from scipy import stats as st
    rows = []
    for m in metrics:
        if m not in env_summary:
            continue
        groups = [g[m].dropna().values for _, g in env_summary.groupby(by)]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        shapiro_ps = [st.shapiro(g).pvalue for g in groups if np.std(g) > 0]
        norm_min_p = min(shapiro_ps) if shapiro_ps else np.nan
        if all(np.std(g) > 0 for g in groups):
            lev_stat, lev_p = st.levene(*groups)
        else:
            lev_stat, lev_p = np.nan, np.nan
        normal_ok = np.isfinite(norm_min_p) and norm_min_p > 0.05
        homo_ok = np.isfinite(lev_p) and lev_p > 0.05
        rows.append({'metric': m, 'shapiro_min_p': norm_min_p,
                     'levene_stat': lev_stat, 'levene_p': lev_p,
                     'normal_ok': normal_ok, 'homoscedastic_ok': homo_ok,
                     'parametric_ok': bool(normal_ok and homo_ok)})
    return pd.DataFrame(rows)


def format_apa(row):
    """Format one compare_environments_stats row as an APA string (ASCII only)."""
    def _p(x):
        if not np.isfinite(x):
            return "p = n/a"
        return "p < .001" if x < .001 else ("p = " + f"{x:.3f}"[1:])

    def _es(x):
        return f"{x:.3f}"[1:] if (np.isfinite(x) and abs(x) < 1) else f"{x:.3f}"

    test = str(row.get('test', ''))
    stat = row.get('statistic', np.nan)
    p = row.get('p_value', np.nan)
    if 'ANOVA' in test:
        s = f"F({int(row['df1'])},{int(row['df2'])}) = {stat:.2f}, {_p(p)}"
        if np.isfinite(row.get('eta_sq', np.nan)):
            s += f", eta^2 = {_es(row['eta_sq'])}"
    elif 'Kruskal' in test:
        s = f"H({int(row['df1'])}) = {stat:.2f}, {_p(p)}"
        if np.isfinite(row.get('eps_sq', np.nan)):
            s += f", eps^2 = {_es(row['eps_sq'])}"
    elif 'Welch' in test:
        s = f"t = {stat:.2f}, {_p(p)}"
        p_adj = row.get('p_adj', np.nan)
        if np.isfinite(p_adj):
            s += ", " + _p(p_adj).replace("p ", "p_adj ")
        if np.isfinite(row.get('cohens_d', np.nan)):
            s += f", d = {row['cohens_d']:.2f}"
    else:
        s = f"stat = {stat:.2f}, {_p(p)}"
    return s


# ── Statistical tests ─────────────────────────────────────────────────────────

def per_run_corr_test(env_metrics, x='dist_nest_mean', y='alpha_mean_window',
                      by='environment', seed_col='seed', boot_seed=0):
    """One Pearson r(x, y) per run, then a one-sample t-test of those r's against 0
    per environment, with a bootstrap 95% CI on the mean r."""
    from scipy import stats as st
    rows = []
    for name, g in env_metrics.groupby(by):
        rs = []
        for _, gs in g.groupby(seed_col):
            a = np.asarray(gs[x].values, dtype=float)
            b = np.asarray(gs[y].values, dtype=float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 2 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
                rs.append(np.corrcoef(a[m], b[m])[0, 1])
        rs = np.asarray(rs)
        t, p = (st.ttest_1samp(rs, 0.0) if len(rs) >= 2 else (np.nan, np.nan))
        _, lo, hi = bootstrap_ci(rs, np.mean, seed=boot_seed)
        rows.append({by: name,
                     'mean_r': rs.mean() if len(rs) else np.nan,
                     'std_r': rs.std(ddof=1) if len(rs) > 1 else np.nan,
                     'ci_low': lo, 'ci_high': hi,
                     't_vs_0': t, 'p_value': p, 'n_runs': len(rs)})
    return pd.DataFrame(rows)


def per_run_role_fractions(agent_df, swarm_df, thresholds=ROLE_THRESHOLDS,
                           spans=(('window', 0.5), ('whole', 0.0))):
    """Per-run role-mix fractions for each span (fraction from the end): timeavg_*
    (population share over steps) and winrole_* (per-agent span-mean alpha)."""
    roles = ('exploiter', 'explorer', 'switcher')
    out = {}
    s_min, s_max = agent_df['step'].min(), agent_df['step'].max()
    for span_name, wf in spans:
        cutoff = s_min + wf * (s_max - s_min)
        sw = swarm_df[swarm_df['step'] >= cutoff] if 'step' in swarm_df else swarm_df
        for role in roles:
            col = f'{role}_fraction'
            out[f'timeavg_{span_name}_{role}_frac'] = (
                float(sw[col].mean()) if col in sw and len(sw) else np.nan)
        win = agent_df[agent_df['step'] >= cutoff]
        amean = win.groupby('agent_id')['alpha'].mean()
        rep = amean.map(lambda a: role_from_alpha(a, thresholds))
        share = rep.value_counts(normalize=True)
        for role in roles:
            out[f'winrole_{span_name}_{role}_frac'] = float(share.get(role, 0.0))
    return out


def compare_environments_stats(env_summary,
                               metrics=('timeavg_window_exploiter_frac',
                                        'mean_shannon_diversity',
                                        'pop_dist_nest_mean',
                                        'pop_exploitation_index'),
                               by='environment'):
    """Between-environment tests on run-level outcomes: per metric, one-way ANOVA
    (eta^2) and Kruskal-Wallis (epsilon^2) omnibus, then pairwise Welch t-tests
    (Cohen's d) with BH-FDR and Holm correction."""
    from scipy import stats as st
    import itertools
    groups = list(dict.fromkeys(env_summary[by]))
    k = len(groups)
    rows = []
    for m in metrics:
        if m not in env_summary:
            continue
        samples = [env_summary.loc[env_summary[by] == g, m].dropna().values
                   for g in groups]
        ns = [len(s) for s in samples]
        if min(ns) < 2:
            continue
        n_total = sum(ns)
        df_between, df_within = k - 1, n_total - k
        F, p = st.f_oneway(*samples)
        rows.append({'metric': m, 'test': 'ANOVA (omnibus)',
                     'statistic': F, 'p_value': p, 'n': n_total,
                     'df1': df_between, 'df2': df_within,
                     'eta_sq': _eta_squared_anova(F, df_between, df_within)})
        H, p = st.kruskal(*samples)
        rows.append({'metric': m, 'test': 'Kruskal-Wallis (omnibus)',
                     'statistic': H, 'p_value': p, 'n': n_total,
                     'df1': df_between, 'eps_sq': _epsilon_squared_kruskal(H, n_total)})
        # pairwise Welch, corrected within this metric family
        pair_rows, pair_ps = [], []
        for a, b in itertools.combinations(range(k), 2):
            t, p = st.ttest_ind(samples[a], samples[b], equal_var=False)
            pair_rows.append({'metric': m,
                              'test': f'Welch t: {groups[a]} vs {groups[b]}',
                              'statistic': t, 'p_value': p, 'n': ns[a] + ns[b],
                              'cohens_d': cohens_d(samples[a], samples[b])})
            pair_ps.append(p)
        if pair_ps:
            valid = np.isfinite(pair_ps)
            p_adj = np.full(len(pair_ps), np.nan)
            p_holm = np.full(len(pair_ps), np.nan)
            reject = np.zeros(len(pair_ps), dtype=bool)
            if valid.any():
                vp = np.asarray(pair_ps)[valid]
                padj_bh = adjust_pvalues(vp, method='fdr_bh')
                pholm = adjust_pvalues(vp, method='holm')
                p_adj[valid] = padj_bh
                p_holm[valid] = pholm
                reject[valid] = padj_bh < 0.05
            for pr, pa, ph, rj in zip(pair_rows, p_adj, p_holm, reject):
                pr.update({'p_adj': pa, 'p_holm': ph, 'reject_fdr': bool(rj)})
        rows.extend(pair_rows)
    return pd.DataFrame(rows)


# ── Behavioural role clustering (observed-behaviour role inference) ────────────
# Infer roles from observed behaviour (mobility / harvest / yield), excluding the
# environment-specific distance-to-nest features.

BEHAVIOR_FEATURES = [
    'gyration_radius',          # home-range tightness (low = anchored exploiter)
    'window_distance',          # path length over the window (mobility)
    'harvest_success_rate',     # fraction of steps with a harvest
    'yield_per_distance',       # delivered resource per unit travelled
    'local_resource_mean',      # richness of the experienced neighbourhood
    'deliveries_per_100steps',  # trip throughput
]

# Sign of each feature in an "exploiter-ness" score (high score => exploiter).
_EXPLOITER_SIGN = {
    'gyration_radius': -1.0, 'window_distance': -1.0,
    'harvest_success_rate': +1.0, 'yield_per_distance': +1.0,
    'local_resource_mean': +1.0, 'deliveries_per_100steps': +1.0,
    'frac_time_near_nest': +1.0, 'time_since_success_mean': -1.0,
}


def _clean_standardize(df, features):
    """z-score the feature matrix, imputing non-finite cells with column medians."""
    from sklearn.preprocessing import StandardScaler
    X = df[features].to_numpy(dtype=float)
    finite = np.where(np.isfinite(X), X, np.nan)
    col_med = np.nanmedian(finite, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    bad = ~np.isfinite(X)
    if bad.any():
        bi = np.where(bad)
        X[bi] = np.take(col_med, bi[1])
    return StandardScaler().fit_transform(X)


def _label_clusters(Xs, labels, features):
    """Map opaque cluster ids to role names by centroid exploiter-ness score."""
    signs = np.array([_EXPLOITER_SIGN.get(f, 0.0) for f in features])
    score = Xs @ signs  # high => exploiter-like
    uniq = list(np.unique(labels))
    cluster_score = {c: float(score[labels == c].mean()) for c in uniq}
    order = sorted(uniq, key=lambda c: cluster_score[c], reverse=True)
    mapping = {}
    for rank, c in enumerate(order):
        if rank == 0:
            mapping[c] = 'exploiter'
        elif rank == len(order) - 1:
            mapping[c] = 'explorer'
        else:
            mapping[c] = 'switcher'
    return np.array([mapping[l] for l in labels])


def cluster_behavioral_roles(agent_metrics, features=None, k=3, method='gmm',
                             random_state=0):
    """Cluster agents in the standardized behaviour space (method='gmm' or
    'kmeans') and name clusters by exploiter-ness; adds 'cluster' and
    'behavioral_role' columns."""
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    df = agent_metrics.reset_index(drop=True).copy()
    features = list(features) if features is not None else list(BEHAVIOR_FEATURES)
    Xs = _clean_standardize(df, features)
    if method == 'kmeans':
        labels = KMeans(n_clusters=k, n_init=10,
                        random_state=random_state).fit_predict(Xs)
    else:
        labels = GaussianMixture(n_components=k, covariance_type='full',
                                 n_init=5, random_state=random_state).fit_predict(Xs)
    df['cluster'] = labels
    df['behavioral_role'] = _label_clusters(Xs, labels, features)
    return df


# Distribution-invariant exploiter signature (distance-to-nest excluded: its sign
# flips across environments).
EXPLOITATION_FEATURES = ('gyration_radius', 'local_resource_mean',
                         'yield_per_distance')


def compute_exploitation_index(df, features=EXPLOITATION_FEATURES, method='pca'):
    """Collapse the standardized exploiter signature into one per-agent index (FEI),
    oriented so high = exploiter. method='pca' (PC1, reports explained variance)
    or 'zscore'. Returns (df + 'exploitation_index', info with loadings)."""
    features = list(features)
    out = df.reset_index(drop=True).copy()
    if out.empty or len(out) < 2:
        out['exploitation_index'] = np.nan
        return out, {'method': method, 'explained_variance_ratio': np.nan,
                     'loadings': {f: np.nan for f in features}}
    Xs = _clean_standardize(out, features)
    signs = np.array([_EXPLOITER_SIGN.get(f, 0.0) for f in features])
    expl_score = Xs @ signs  # reference "exploiter-ness" direction (high => exploiter)
    if method == 'zscore':
        idx = expl_score.copy()
        loadings = {f: float(s) for f, s in zip(features, signs)}
        evr = np.nan
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1, random_state=0)
        idx = pca.fit_transform(Xs)[:, 0]
        evr = float(pca.explained_variance_ratio_[0])
        comp = pca.components_[0]
        if np.corrcoef(idx, expl_score)[0, 1] < 0:  # orient: high => exploiter
            idx, comp = -idx, -comp
        loadings = {f: float(c) for f, c in zip(features, comp)}
    out['exploitation_index'] = idx
    return out, {'method': method, 'explained_variance_ratio': evr,
                 'loadings': loadings}


# ── Parameter sensitivity (one-at-a-time) ──────────────────────────────────────

# The five hand-tuned adaptation/steering constants swept by the OAT analysis.
OAT_PARAMS = ('failure_streak', 'alpha_learn_rate', 'reward_scale',
              'nest_pull', 'crowd_avoid')
OAT_OUTCOMES = ('corr_fei_alpha', 'timeavg_window_exploiter_frac',
                'mean_shannon_diversity', 'corr_distnest_alpha')


def oat_sensitivity_summary(sweep_df, param, outcomes=OAT_OUTCOMES,
                            seed_col='seed'):
    """Mean +/- std of each outcome at each value of `param`, across seeds;
    returns <outcome>_mean / <outcome>_std columns per parameter value."""
    cols = [c for c in outcomes if c in sweep_df.columns]
    g = sweep_df.groupby(param)[cols]
    summary = g.agg(['mean', 'std'])
    summary.columns = [f'{c}_{stat}' for c, stat in summary.columns]
    return summary.reset_index()


def plot_oat_sensitivity(sweep_df, params=OAT_PARAMS, outcomes=OAT_OUTCOMES,
                         baseline=None):
    """Small-multiples grid (rows=constants, cols=outcomes) of outcome mean +/- std
    vs the swept value, with an optional baseline line per panel."""
    outcomes = [c for c in outcomes if c in sweep_df.columns]
    params = [p for p in params if p in sweep_df.columns
              and sweep_df[p].nunique() > 1]
    nr, nc = len(params), len(outcomes)
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.6 * nr),
                             squeeze=False)
    for i, p in enumerate(params):
        sub = sweep_df[sweep_df[p].notna()]
        s = oat_sensitivity_summary(sub, p, outcomes=outcomes)
        for j, out in enumerate(outcomes):
            ax = axes[i][j]
            if f'{out}_mean' in s:
                ax.errorbar(s[p], s[f'{out}_mean'], yerr=s[f'{out}_std'],
                            marker='o', capsize=3)
            if baseline and p in baseline:
                ax.axvline(baseline[p], color='gray', linestyle=':')
            if out == 'corr_distnest_alpha':
                ax.axhline(0, color='red', linewidth=0.6)
            if i == 0:
                ax.set_title(out, fontsize=8)
            if j == 0:
                ax.set_ylabel(p, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle('One-at-a-time parameter sensitivity', fontsize=12)
    fig.tight_layout()
    return fig
