import numpy as np
import math


class ResourceItem:
    def __init__(self, x, y, max_amount=1.0, regen_cooldown_steps=10):
        self.x = x
        self.y = y
        self.max_amount = max_amount
        self.current_amount = max_amount
        self.regen_cooldown_steps = regen_cooldown_steps
        self.depleted_cooldown = 0
        self.consumed_this_step = 0.0

    def regenerate(self):
        self.consumed_this_step = 0.0
        if self.depleted_cooldown > 0:
            self.depleted_cooldown -= 1
            if self.depleted_cooldown == 0:
                self.current_amount = self.max_amount

    def mark_depleted(self):
        self.current_amount = 0.0
        self.depleted_cooldown = self.regen_cooldown_steps

    @property
    def is_available(self):
        return self.depleted_cooldown == 0 and self.current_amount > 0.01


class Nest:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.total_deposits = 0.0
        self.deposit_history = []

    def record_deposit(self, step, amount):
        self.total_deposits += amount
        self.deposit_history.append((step, amount))


class ResourcePatch:
    """One circular patch of ResourceItems. cooldown = int(1/depletion_rate);
    distribution sets the radial density profile."""

    def __init__(self, patch_id, x, y, radius,
                 n_items=150, item_max_amount=1.0,
                 depletion_rate=0.07, distribution='gaussian', rng=None,
                 bounds=None):
        self.id = patch_id
        self.x = x
        self.y = y
        self.radius = radius
        self.n_items = n_items
        self.item_max_amount = item_max_amount
        self.depletion_rate = depletion_rate
        self.regen_cooldown_steps = max(1, int(1.0 / depletion_rate))
        self.distribution = distribution
        # (width, height); when set, item coordinates are clamped inside the arena.
        self.bounds = bounds
        self.occupancy = 0
        self.items = []
        self._rng = rng if rng is not None else np.random.default_rng()
        self._generate_items()

    def _generate_items(self):
        for _ in range(self.n_items):
            if self.distribution == 'gaussian':
                r = abs(self._rng.normal(0, self.radius / 3))
                r = min(r, self.radius)
            elif self.distribution == 'power_law':
                r = self.radius * (1 - self._rng.power(2))
            elif self.distribution == 'ring':
                # Tight annulus near the map edge (r ~ 85% of radius ± 5%)
                r = self._rng.normal(0.85 * self.radius, 0.05 * self.radius)
                r = float(np.clip(r, 0.0, self.radius))
            elif self.distribution == 'donut':
                # Smooth bell peaked at mid-radius — exploration cost is graded
                r = self._rng.normal(0.55 * self.radius, 0.10 * self.radius)
                r = float(np.clip(r, 0.0, self.radius))
            elif self.distribution == 'gradient_out':
                # Density rises with distance from center: sample r = R*(1-U^2) → dense at R
                r = self.radius * (1.0 - self._rng.uniform(0, 1) ** 2)
            elif self.distribution == 'gradient_in':
                # Density falls with distance from center: sample r = R*U^2 → dense at 0
                r = self.radius * self._rng.uniform(0, 1) ** 2
            else:  # uniform
                r = self.radius * math.sqrt(self._rng.uniform(0, 1))
            angle = self._rng.uniform(0, 2 * math.pi)
            ix = float(self.x + r * math.cos(angle))
            iy = float(self.y + r * math.sin(angle))
            if self.bounds is not None:
                ix = min(max(ix, 0.0), self.bounds[0])
                iy = min(max(iy, 0.0), self.bounds[1])
            self.items.append(
                ResourceItem(ix, iy, self.item_max_amount, self.regen_cooldown_steps)
            )

    def regenerate(self):
        self.occupancy = 0
        for item in self.items:
            item.regenerate()

    @property
    def current_resource(self):
        return sum(item.current_amount for item in self.items)

    @property
    def max_capacity(self):
        return self.n_items * self.item_max_amount

    @property
    def consumed_this_step(self):
        return sum(item.consumed_this_step for item in self.items)

    @property
    def available_items(self):
        return sum(1 for item in self.items if item.is_available)

    @property
    def resource_fraction(self):
        mc = self.max_capacity
        return self.current_resource / mc if mc > 0 else 0.0


class EnvironmentEventManager:
    """Resource crisis: below CRISIS_THRESHOLD capacity, boost pheromone decay in
    the region; resolves above RECOVERY_THRESHOLD."""
    CRISIS_THRESHOLD = 0.20
    RECOVERY_THRESHOLD = 0.40
    DECAY_BOOST = 0.90  # extra multiplier applied per step in crisis region

    def __init__(self):
        self.crisis_patches = set()

    @property
    def crisis_active(self):
        return len(self.crisis_patches) > 0

    def check_events(self, patches, env):
        for patch in patches:
            pid = patch.id
            frac = patch.resource_fraction
            if pid in self.crisis_patches:
                if frac >= self.RECOVERY_THRESHOLD:
                    self.crisis_patches.discard(pid)
                else:
                    self._apply_decay_boost(patch, env)
            else:
                if frac < self.CRISIS_THRESHOLD:
                    self.crisis_patches.add(pid)
                    self._apply_decay_boost(patch, env)

    def _apply_decay_boost(self, patch, env):
        if not env.pheromones_enabled:
            return
        gx_idx = np.arange(env.grid_size[0])
        gy_idx = np.arange(env.grid_size[1])
        gx, gy = np.meshgrid(gx_idx, gy_idx, indexing='ij')
        dist = np.hypot(gx * env.grid_res - patch.x, gy * env.grid_res - patch.y)
        mask = dist <= patch.radius
        env.pheromones[mask] *= self.DECAY_BOOST


class Environment:
    COLLECTION_RADIUS = 2.0

    def __init__(self, width, height, grid_res=2.0, pheromones_enabled=True):
        self.width = width
        self.height = height
        self.grid_res = grid_res
        self.pheromones_enabled = pheromones_enabled
        self.grid_size = (int(width / grid_res) + 1, int(height / grid_res) + 1)
        self.pheromones = np.zeros(self.grid_size)
        self.resource_grid = np.zeros(self.grid_size)  # fast heuristic cache
        self.patches = []
        self.nest = Nest(50.0, 50.0)
        self.event_manager = EnvironmentEventManager()

    def add_patch(self, patch):
        self.patches.append(patch)

    def update_resource_grid(self):
        """Rebuild resource density grid; called once per step before agent loop."""
        self.resource_grid[:] = 0.0
        for p in self.patches:
            for item in p.items:
                if not item.is_available:
                    continue
                gx = int(item.x / self.grid_res)
                gy = int(item.y / self.grid_res)
                if 0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1]:
                    self.resource_grid[gx, gy] += item.current_amount

    def deposit_pheromone(self, x, y, amount):
        if not self.pheromones_enabled:
            return
        gx = int(x / self.grid_res)
        gy = int(y / self.grid_res)
        if 0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1]:
            self.pheromones[gx, gy] = min(10.0, self.pheromones[gx, gy] + amount)

    def update_pheromones(self, decay_rate=0.98):
        if self.pheromones_enabled:
            self.pheromones *= decay_rate

    def get_pheromone_at_angle(self, x, y, angle, radius):
        """τ_j: pheromone at probe point (x,y) + radius in direction angle."""
        if not self.pheromones_enabled:
            return 0.0
        gx = int((x + math.cos(angle) * radius) / self.grid_res)
        gy = int((y + math.sin(angle) * radius) / self.grid_res)
        if 0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1]:
            return float(self.pheromones[gx, gy])
        return 0.0

    def get_resource_heuristic_at_angle(self, x, y, angle, radius):
        """η_j: resource density at probe point (grid lookup, O(1))."""
        gx = int((x + math.cos(angle) * radius) / self.grid_res)
        gy = int((y + math.sin(angle) * radius) / self.grid_res)
        if 0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1]:
            return max(0.01, float(self.resource_grid[gx, gy]))
        return 0.01

    def get_local_resources(self, x, y, sensing_radius):
        total = 0.0
        for p in self.patches:
            if math.hypot(p.x - x, p.y - y) > sensing_radius + p.radius:
                continue
            for item in p.items:
                if not item.is_available:
                    continue
                dist = math.hypot(item.x - x, item.y - y)
                if dist <= sensing_radius:
                    total += item.current_amount * max(0.0, 1.0 - dist / sensing_radius)
        return total

    def consume_resources(self, x, y, amount, sensing_radius):
        collected = 0.0
        for p in self.patches:
            patch_dist = math.hypot(p.x - x, p.y - y)
            if patch_dist <= p.radius:
                p.occupancy += 1
            if patch_dist > p.radius + self.COLLECTION_RADIUS:
                continue
            for item in p.items:
                if not item.is_available:
                    continue
                dist = math.hypot(item.x - x, item.y - y)
                if dist <= self.COLLECTION_RADIUS:
                    take = min(amount - collected, item.current_amount)
                    if take <= 0:
                        continue
                    item.consumed_this_step += take
                    collected += take
                    item.mark_depleted()
                    if collected >= amount:
                        return collected
        return collected

    def deposit_resources_at_nest(self, x, y, amount, step=None):
        if amount <= 0.01:
            return 0.0, False
        if math.hypot(self.nest.x - x, self.nest.y - y) <= self.COLLECTION_RADIUS:
            self.nest.record_deposit(step=step, amount=amount)
            return amount, True
        return 0.0, False

    def step(self):
        self.update_pheromones()
        for p in self.patches:
            p.regenerate()
        self.event_manager.check_events(self.patches, self)
        return self.event_manager.crisis_active
