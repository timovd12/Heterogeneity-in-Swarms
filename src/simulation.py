import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.agents import (
    Agent, FAILURE_STREAK, ALPHA_LEARN_RATE, REWARD_SCALE, NEST_PULL, CROWD_AVOID,
)
from src.environment import Environment, ResourcePatch


class Simulation:
    """Main simulation: arena-spanning resource field, ACO agents, trail
    pheromones, crisis events. exploiter_fraction is deprecated (agents start
    homogeneous at α=β=0.5; roles emerge)."""

    def __init__(
        self,
        num_agents=100,
        width=100.0,
        height=100.0,
        patch_distribution='gaussian',
        pheromones_enabled=True,
        seed=None,
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
        self.width = width
        self.height = height
        self.pheromones_enabled = pheromones_enabled
        self.crowding_sensitivity = crowding_sensitivity
        self.pheromone_weight = pheromone_weight
        self.energy_return_threshold = energy_return_threshold
        self.sensing_radius = sensing_radius
        self.pheromone_sensing_radius = (
            pheromone_sensing_radius if pheromone_sensing_radius is not None
            else sensing_radius
        )
        self.rng = np.random.default_rng(seed)

        self.env = Environment(width, height, pheromones_enabled=pheromones_enabled)
        # Single centered patch spanning the arena diagonal; items clamped to bounds.
        env_radius = math.hypot(width, height) / 2.0
        self.env.add_patch(ResourcePatch(
            0,
            width / 2, height / 2,
            radius=env_radius,
            n_items=n_items,
            item_max_amount=1.0,
            depletion_rate=depletion_rate,
            distribution=patch_distribution,
            rng=self.rng,
            bounds=(width, height),
        ))

        # Homogeneous swarm: every agent starts identical (α=β=0.5, full energy,
        # spawned at the nest); roles emerge rather than being seeded.
        self.agents = []
        for i in range(num_agents):
            self.agents.append(Agent(
                i,
                self.env.nest.x,
                self.env.nest.y,
                self.env,
                alpha=0.5,
                beta=0.5,
                pheromones_enabled=pheromones_enabled,
                crowding_sensitivity=crowding_sensitivity,
                sensing_radius=sensing_radius,
                pheromone_sensing_radius=self.pheromone_sensing_radius,
                pheromone_weight=pheromone_weight,
                energy_return_threshold=energy_return_threshold,
                failure_streak=failure_streak,
                alpha_learn_rate=alpha_learn_rate,
                reward_scale=reward_scale,
                nest_pull=nest_pull,
                crowd_avoid=crowd_avoid,
                rng=self.rng,
            ))

        self.agent_logs = []
        self.env_logs = []
        self.swarm_logs = []
        self.role_switch_logs = []
        self._visited_cells = set()

    def _track_coverage(self):
        for a in self.agents:
            self._visited_cells.add(
                (int(a.x / self.env.grid_res), int(a.y / self.env.grid_res))
            )

    def step(self, step_idx):
        # Snapshot resource grid once per step (O(n_items), fast heuristic lookups)
        self.env.update_resource_grid()

        positions = np.array([(a.x, a.y) for a in self.agents])
        total_deposited = 0.0
        total_harvested_local = 0.0
        n = len(self.agents)
        role_counts = {'exploiter': 0, 'explorer': 0, 'switcher': 0}
        n_switches_this_step = 0

        # Per-agent crowding count + direction away from the neighbour centroid.
        crowding_counts = np.empty(n, dtype=int)
        crowd_away_angles = [None] * n
        for idx, a in enumerate(self.agents):
            dx = positions[:, 0] - a.x
            dy = positions[:, 1] - a.y
            d = np.hypot(dx, dy)
            mask = (d < a.sensing_radius) & (d > 1e-9)
            cnt = int(mask.sum())
            crowding_counts[idx] = cnt
            if cnt > 0:
                cx = positions[mask, 0].mean()
                cy = positions[mask, 1].mean()
                # Angle pointing from the neighbour centroid toward the agent.
                if math.hypot(a.x - cx, a.y - cy) > 1e-9:
                    crowd_away_angles[idx] = math.atan2(a.y - cy, a.x - cx)
        mean_crowding = float(crowding_counts.mean()) if n else 1.0

        for idx, agent in enumerate(self.agents):
            nearby = int(crowding_counts[idx])
            agent.swarm_mean_crowding = mean_crowding
            agent.crowd_away_angle = crowd_away_angles[idx]
            held_before = agent.held_resources
            prev_role = agent.current_role

            agent.sense_and_act(nearby)

            harvested = self.env.consume_resources(
                agent.x, agent.y, agent.harvest_rate, agent.sensing_radius
            )
            agent.held_resources = agent.held_resources + harvested  # no carry cap
            total_harvested_local += harvested

            deposited, success = self.env.deposit_resources_at_nest(
                agent.x, agent.y, agent.held_resources, step_idx
            )
            if success:
                agent.held_resources = 0.0
                total_deposited += deposited

            agent.apply_step_outcome(harvested, deposited)
            agent._update_energy(deposited_amount=deposited)

            if agent.current_role != prev_role:
                n_switches_this_step += 1
                self.role_switch_logs.append({
                    'step': step_idx,
                    'agent_id': agent.id,
                    'old_role': prev_role,
                    'new_role': agent.current_role,
                    'role_switches_total': agent.role_switches,
                    'alpha': agent.alpha,
                    'beta': agent.beta,
                })

            role_counts[agent.current_role] = role_counts.get(agent.current_role, 0) + 1

            self.agent_logs.append({
                'step': step_idx,
                'agent_id': agent.id,
                'x': agent.x,
                'y': agent.y,
                'local_resource_level': agent.local_resource_level,
                'nearby_agents': agent.local_crowding,
                'harvested_resources': harvested,
                'deposited_resources': deposited,
                'speed': agent.speed,
                'energy': agent.energy,
                'alpha': agent.alpha,
                'beta': agent.beta,
                'search_difficulty': agent.search_difficulty,
                'role': agent.current_role,
                'role_duration': agent.role_duration,
                'role_switches': agent.role_switches,
                'time_since_last_success': agent.time_since_last_success,
                'held_resources': held_before if success else agent.held_resources,
                'is_returning_to_nest': agent.is_returning_to_nest,
                'deposits_this_step': 1.0 if success else 0.0,
                'distance_traveled': agent.distance_traveled,
                'sensing_radius': agent.sensing_radius,
                'pheromone_sensing_radius': agent.pheromone_sensing_radius,
            })

        self._track_coverage()
        crisis_active = self.env.step()

        for patch in self.env.patches:
            self.env_logs.append({
                'step': step_idx,
                'patch_id': patch.id,
                'resource_level': patch.current_resource,
                'resource_fraction': patch.resource_fraction,
                'patch_occupancy': patch.occupancy,
                'consumed_this_step': patch.consumed_this_step,
                'available_items': patch.available_items,
                'total_items': patch.n_items,
                'crisis_active': crisis_active,
            })

        dispersion = float(np.mean(
            np.linalg.norm(positions - positions.mean(axis=0), axis=1)
        ))

        self.swarm_logs.append({
            'step': step_idx,
            'total_resource_collected': total_deposited,
            'total_harvested_local': total_harvested_local,
            'spatial_dispersion': dispersion,
            'mean_energy': float(np.mean([a.energy for a in self.agents])),
            'min_energy': float(np.min([a.energy for a in self.agents])),
            'max_energy': float(np.max([a.energy for a in self.agents])),
            'exploiter_fraction': role_counts.get('exploiter', 0) / n,
            'explorer_fraction': role_counts.get('explorer', 0) / n,
            'switcher_fraction': role_counts.get('switcher', 0) / n,
            'n_role_switches_this_step': n_switches_this_step,
            'mean_held_resources': float(np.mean([a.held_resources for a in self.agents])),
            'agents_returning': sum(1 for a in self.agents if a.is_returning_to_nest),
            'total_deposits_this_step': total_deposited,
            'total_distance': sum(a.distance_traveled for a in self.agents),
            'coverage_fraction': len(self._visited_cells) / max(
                1, self.env.grid_size[0] * self.env.grid_size[1]
            ),
            'mean_alpha': float(np.mean([a.alpha for a in self.agents])),
            'mean_beta': float(np.mean([a.beta for a in self.agents])),
            'crisis_active': crisis_active,
        })

    def run(self, steps, render_callback=None):
        for i in range(steps):
            self.step(i)
            if render_callback is not None:
                if render_callback(self, i) is False:
                    break
        return self._to_dataframes()

    def run_live(self, steps, viewer_config=None):
        from src.viewer import LiveViewer
        viewer = LiveViewer(self, config=viewer_config or {})
        viewer.total_steps = steps
        try:
            result = self.run(steps, render_callback=viewer.render_frame)
            viewer.wait_for_close()
            return result
        finally:
            viewer.close()

    def save_logs(self, output_dir=None):
        """Save all DataFrames as CSVs under data/logs/ (or a custom path)."""
        base = (
            Path(output_dir) if output_dir
            else Path(__file__).parent.parent / 'data' / 'logs'
        )
        agent_df, env_df, swarm_df, switch_df = self._to_dataframes()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        for sub in ('agents', 'environment', 'swarm', 'switches'):
            (base / sub).mkdir(parents=True, exist_ok=True)
        agent_df.to_csv(base / 'agents'      / f'agent_logs_{ts}.csv',       index=False)
        env_df.to_csv(base   / 'environment' / f'environment_logs_{ts}.csv', index=False)
        swarm_df.to_csv(base / 'swarm'       / f'swarm_logs_{ts}.csv',       index=False)
        switch_df.to_csv(base / 'switches'   / f'role_switches_{ts}.csv',    index=False)
        print(f"Logs saved to {base}/")
        return base

    def _to_dataframes(self):
        agent_df  = pd.DataFrame(self.agent_logs)
        env_df    = pd.DataFrame(self.env_logs)
        swarm_df  = pd.DataFrame(self.swarm_logs)
        switch_df = (
            pd.DataFrame(self.role_switch_logs)
            if self.role_switch_logs
            else pd.DataFrame(columns=[
                'step', 'agent_id', 'old_role', 'new_role',
                'role_switches_total', 'alpha', 'beta',
            ])
        )
        return agent_df, env_df, swarm_df, switch_df
