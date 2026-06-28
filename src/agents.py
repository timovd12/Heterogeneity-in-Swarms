import numpy as np
import math

RESOURCE_SCALE    = 1.0   # local resource level that saturates the resource exploit drive
REWARD_SCALE      = 0.3   # recent-reward EMA that saturates the success exploit drive
REWARD_EMA        = 0.1   # weight of the newest outcome in the recent-reward EMA
FAILURE_STREAK    = 20    # steps without success that trigger the explore push
ALPHA_LEARN_RATE  = 0.02  # step size for α/β adaptation
NEST_PULL         = 5.0   # strength of the homeward urgency at zero energy
CROWD_AVOID       = 4.0   # strength of steering away from crowded directions
ENERGY_CARRY_COST = 0.02  # extra energy/step per unit carried


class Agent:
    """Forager with ACO direction choice P_j ∝ τ_j^α·η_j^β. α (exploit) and β
    (explore) sum to 1 and adapt from foraging success; role follows α
    (exploiter >0.6, explorer <0.4, switcher in between)."""

    ENERGY_MAX              = 100.0
    ENERGY_MOVE_COST        = 0.08
    ENERGY_IDLE_COST        = 0.02
    ENERGY_HARVEST_GAIN     = 8.0
    ENERGY_LOW_THRESHOLD    = 10.0   # below this: speed capped at 0.3
    ENERGY_RETURN_THRESHOLD = 40.0   # below this: homeward urgency begins

    def __init__(
        self,
        agent_id,
        x,
        y,
        env,
        alpha=0.5,
        beta=0.5,
        pheromones_enabled=True,
        crowding_sensitivity=1.0,
        sensing_radius=6.0,
        pheromone_sensing_radius=None,
        pheromone_weight=1.0,
        energy_return_threshold=None,
        failure_streak=FAILURE_STREAK,
        alpha_learn_rate=ALPHA_LEARN_RATE,
        reward_scale=REWARD_SCALE,
        nest_pull=NEST_PULL,
        crowd_avoid=CROWD_AVOID,
        rng=None,
    ):
        if energy_return_threshold is not None:
            self.ENERGY_RETURN_THRESHOLD = float(energy_return_threshold)
        # Per-instance adaptation/steering constants for the OAT sweep.
        self.failure_streak   = float(failure_streak)
        self.alpha_learn_rate = float(alpha_learn_rate)
        self.reward_scale     = float(reward_scale)
        self.nest_pull        = float(nest_pull)
        self.crowd_avoid      = float(crowd_avoid)
        # Seeded generator (from the Simulation) for reproducible movement.
        self.rng = rng if rng is not None else np.random.default_rng()
        self.id = agent_id
        self.x = float(x)
        self.y = float(y)
        self.env = env
        self.pheromones_enabled = pheromones_enabled
        # Scales the τ exponent: 1.0 = full trail-following, 0.0 = ignore trails.
        self.pheromone_weight = float(pheromone_weight)
        self.heading = 0.0   # identical start: divergence comes from dynamics, not state
        self.speed = 1.0
        self.sensing_radius = sensing_radius
        self.pheromone_sensing_radius = (
            pheromone_sensing_radius if pheromone_sensing_radius is not None
            else sensing_radius
        )
        self.harvest_rate = 0.75
        self.crowding_sensitivity = crowding_sensitivity

        # Normalise so α + β = 1
        total = float(alpha) + float(beta)
        self.alpha = float(alpha) / total if total > 0 else 0.5
        self.beta  = float(beta)  / total if total > 0 else 0.5

        self.energy = self.ENERGY_MAX   # identical full energy for every agent
        self.held_resources = 0.0
        self.is_returning_to_nest = False

        self.time_since_last_success = 0
        self.recent_reward = 0.0          # EMA of collected resources
        self.local_crowding = 0
        self.swarm_mean_crowding = 1.0     # set by the simulation each step
        self.crowd_away_angle = None       # direction away from neighbours (set by sim)
        self.distance_traveled = 0.0
        self.search_difficulty = 0.0
        self.harvest_this_step = 0.0
        self.local_resource_level = 0.0

        self.current_role = self._infer_role()
        self.role_duration = 1
        self.role_switches = 0

    # ── Role inference ────────────────────────────────────────────────────────

    def _infer_role(self) -> str:
        if self.alpha > 0.6:
            return 'exploiter'
        if self.alpha < 0.4:
            return 'explorer'
        return 'switcher'

    # ── ACO direction choice ──────────────────────────────────────────────────

    def _choose_direction(self) -> float:
        """Sample a heading from P_j ∝ τ_j^(w·α)·η_j^β·ν_j^(NEST_PULL·u)·ρ_j^avoid:
        τ=pheromone, η=resource heuristic, ν=toward-nest (urgency u), ρ=away-from-crowd."""
        directions = np.linspace(0, 2 * math.pi, 16, endpoint=False)

        eta = np.array([
            self.env.get_resource_heuristic_at_angle(
                self.x, self.y, d, self.sensing_radius
            )
            for d in directions
        ], dtype=float)
        eta = np.clip(eta, 1e-10, None)

        if self.pheromones_enabled:
            tau = np.array([
                self.env.get_pheromone_at_angle(
                    self.x, self.y, d, self.pheromone_sensing_radius
                )
                for d in directions
            ], dtype=float)
            tau = np.clip(tau, 1e-6, None)
            scores = (tau ** (self.pheromone_weight * self.alpha)) * (eta ** self.beta)
        else:
            scores = eta ** self.beta

        # Homeward urgency: 0 at full energy, 1 at empty.
        urgency = max(0.0, 1.0 - self.energy / self.ENERGY_RETURN_THRESHOLD)
        if urgency > 0.0:
            theta_nest = math.atan2(self.env.nest.y - self.y, self.env.nest.x - self.x)
            nu = 0.5 * (1.0 + np.cos(directions - theta_nest))  # 1 toward nest, 0 away
            nu = np.clip(nu, 1e-6, None)
            scores = scores * (nu ** (self.nest_pull * urgency))

        # Crowding avoidance; suppressed for tired agents (they push home to deposit).
        if self.crowd_away_angle is not None:
            crowd_excess = max(0.0, self.local_crowding / max(self.swarm_mean_crowding, 1.0) - 1.0)
            avoid_strength = (
                self.crowd_avoid * crowd_excess * self.crowding_sensitivity * (1.0 - urgency)
            )
            if avoid_strength > 0.0:
                rho = 0.5 * (1.0 + np.cos(directions - self.crowd_away_angle))  # 1 away
                rho = np.clip(rho, 1e-6, None)
                scores = scores * (rho ** avoid_strength)

        scores = np.clip(scores, 1e-10, None)
        probs = scores / scores.sum()
        return float(self.rng.choice(directions, p=probs))

    # ── α / β update rules ────────────────────────────────────────────────────

    def _update_alpha_beta(self, just_harvested: bool):
        """Adapt α/β from success only, then renormalise (α+β=1): rich resources and
        high recent reward push ↑α; a long failure streak pushes ↑β. Crowding is
        excluded on purpose (it steers movement, not the role signal)."""
        res_drive    = min(1.0, self.local_resource_level / RESOURCE_SCALE)
        reward_drive = min(1.0, self.recent_reward / self.reward_scale)

        exploit_drive = 0.5 * res_drive + reward_drive
        explore_drive = (1.0 if self.time_since_last_success > self.failure_streak else 0.0)

        delta = self.alpha_learn_rate * (exploit_drive - explore_drive)
        self.alpha = float(np.clip(self.alpha + delta, 0.1, 0.9))
        self.beta  = float(np.clip(self.beta  - delta, 0.1, 0.9))

        total = self.alpha + self.beta
        self.alpha /= total
        self.beta  /= total

    # ── Energy ────────────────────────────────────────────────────────────────

    def _update_energy(self, deposited_amount=0.0):
        self.search_difficulty = min(1.0, self.time_since_last_success / 30.0)
        cost = (
            self.ENERGY_IDLE_COST
            + self.ENERGY_MOVE_COST * self.speed
            + 0.05 * self.search_difficulty            # harder search costs more
            + ENERGY_CARRY_COST * self.held_resources  # carrying drains faster
        )
        gain = self.ENERGY_HARVEST_GAIN * deposited_amount
        self.energy = float(np.clip(self.energy - cost + gain, 0.0, self.ENERGY_MAX))

    # ── Step callbacks ────────────────────────────────────────────────────────

    def apply_step_outcome(self, harvested: float, deposited: float):
        """Called after harvest/deposit; updates success timer, pheromones, α/β, role."""
        self.harvest_this_step = harvested
        just_harvested = harvested > 0 or deposited > 0

        # Recent-reward EMA, normalised so a full harvest/deposit step ≈ 1.0.
        outcome = min(1.0, (harvested + deposited) / self.harvest_rate)
        self.recent_reward = (1.0 - REWARD_EMA) * self.recent_reward + REWARD_EMA * outcome

        if just_harvested:
            self.time_since_last_success = 0
            # Pheromone laid only while carrying, tying trails to food→nest trips.
            if self.pheromones_enabled and self.held_resources > 0.01:
                self.env.deposit_pheromone(self.x, self.y, 0.5 + 0.5 * harvested)
        else:
            self.time_since_last_success += 1

        self._update_alpha_beta(just_harvested)

        new_role = self._infer_role()
        if new_role != self.current_role:
            self.role_switches += 1
            self.current_role = new_role
            self.role_duration = 1
        else:
            self.role_duration += 1

    def sense_and_act(self, nearby_agents: int):
        """Move one step. Energy urgency (not carry load) drives the return trip;
        speed is capped at low energy. Pheromone is laid only while carrying."""
        self.local_crowding = nearby_agents
        prev_x, prev_y = self.x, self.y

        urgency = max(0.0, 1.0 - self.energy / self.ENERGY_RETURN_THRESHOLD)
        carrying = self.held_resources > 0.01
        self.is_returning_to_nest = (urgency > 0.1) or carrying

        if carrying and self.pheromones_enabled:
            self.env.deposit_pheromone(self.x, self.y, 0.2)

        self.local_resource_level = self.env.get_local_resources(
            self.x, self.y, self.sensing_radius
        )

        chosen = self._choose_direction()
        angle_diff = (chosen - self.heading + math.pi) % (2 * math.pi) - math.pi
        self.heading += 0.4 * angle_diff
        self.heading += self.rng.uniform(-0.3, 0.3) * (1.0 - 0.7 * urgency)  # less noise when tired

        if self.energy < self.ENERGY_LOW_THRESHOLD:
            self.speed = 0.3
        else:
            # Blend forage speed (faster via high β) with return speed as urgency rises.
            forage_target = 0.8 + 0.8 * self.beta
            return_target = 2.0 * max(0.5, self.energy / self.ENERGY_MAX)
            target = (1.0 - urgency) * forage_target + urgency * return_target
            self.speed = float(np.clip(0.7 * self.speed + 0.3 * target, 0.3, 2.2))

        self.x += math.cos(self.heading) * self.speed
        self.y += math.sin(self.heading) * self.speed
        self.x = float(np.clip(self.x, 0.0, self.env.width - 1))
        self.y = float(np.clip(self.y, 0.0, self.env.height - 1))
        self.distance_traveled += math.hypot(self.x - prev_x, self.y - prev_y)
