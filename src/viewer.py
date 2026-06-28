"""
Live Pygame viewer for swarm simulation.
"""

import time
import pygame
from src.analysis import ROLE_PALETTE


class LiveViewer:
  def __init__(self, simulation, config=None):
    cfg = {
      'window_size': 800,
      'fps': 30,
      'show_pheromones': True,
      'agent_radius': 3,
      'auto_close_seconds': 3,
    }
    if config:
      cfg.update(config)
    self.sim = simulation
    self.cfg = cfg
    self.size = cfg['window_size']
    self.scale = self.size / max(simulation.width, simulation.height)
    self.paused = False
    self.speed_mult = 1.0
    self._clock = None
    self._screen = None
    self._font = None
    self._initialized = False
    self.total_steps = 0
    self.finished = False
    self._last_step_idx = 0

  def _init_pygame(self):
    if self._initialized:
      return
    pygame.init()
    pygame.display.set_caption('Swarm ABM — Emergent Roles')
    self._screen = pygame.display.set_mode((self.size, self.size))
    self._clock = pygame.time.Clock()
    self._font = pygame.font.SysFont('consolas', 14)
    self._initialized = True

  def _world_to_screen(self, x, y):
    return int(x * self.scale), int(y * self.scale)

  def _handle_events(self):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return False
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          return False
        if event.key == pygame.K_SPACE:
          self.paused = not self.paused
        if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
          self.speed_mult = min(8.0, self.speed_mult * 1.5)
        if event.key == pygame.K_MINUS:
          self.speed_mult = max(0.25, self.speed_mult / 1.5)
    return True

  def _draw_pheromones(self, env):
    if not self.cfg['show_pheromones'] or not env.pheromones_enabled:
      return
    max_val = float(env.pheromones.max()) if env.pheromones.size else 1.0
    if max_val < 0.01:
      return
    surf = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
    gs = env.grid_size
    cell_px = max(1, int(self.scale * env.grid_res))
    for gx in range(gs[0]):
      for gy in range(gs[1]):
        val = env.pheromones[gx, gy] / max_val
        if val < 0.05:
          continue
        sx, sy = self._world_to_screen(gx * env.grid_res, gy * env.grid_res)
        alpha = int(40 + 80 * val)
        pygame.draw.rect(surf, (180, 140, 60, alpha), (sx, sy, cell_px, cell_px))
    self._screen.blit(surf, (0, 0))

  def _draw_environment(self):
    env = self.sim.env
    self._screen.fill((28, 32, 28))
    pygame.draw.rect(self._screen, (80, 90, 80), (0, 0, self.size, self.size), 2)

    for patch in env.patches:
      cx, cy = self._world_to_screen(patch.x, patch.y)
      pr = int(patch.radius * self.scale)
      pygame.draw.circle(self._screen, (45, 75, 45), (cx, cy), pr, 1)
      for item in patch.items:
        if not item.is_available:
          continue
        ix, iy = self._world_to_screen(item.x, item.y)
        pygame.draw.circle(self._screen, (90, 200, 90), (ix, iy), 2)

    nx, ny = self._world_to_screen(env.nest.x, env.nest.y)
    pygame.draw.circle(self._screen, (220, 180, 60), (nx, ny), 8)
    pygame.draw.circle(self._screen, (180, 140, 40), (nx, ny), 8, 2)

  def _display_role(self, agent):
    # Real alpha-based role, matching the logged fractions.
    return agent.current_role

  def _draw_agents(self):
    r = self.cfg['agent_radius']
    for agent in self.sim.agents:
      sx, sy = self._world_to_screen(agent.x, agent.y)
      color_hex = ROLE_PALETTE.get(self._display_role(agent), '#7f7f7f')
      color = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
      pygame.draw.circle(self._screen, color, (sx, sy), r)

  def _draw_hud(self, step_idx, extra_lines=None):
    role_counts = {}
    for a in self.sim.agents:
      r = self._display_role(a)
      role_counts[r] = role_counts.get(r, 0) + 1
    lines = [
      f'Step {step_idx}  |  FPS target {self.cfg["fps"]}  x{self.speed_mult:.1f}',
      'Space=pause  +/-=speed  Esc=quit',
    ]
    if self.finished:
      lines.append('Simulation complete')
    if self.paused:
      lines.append('** PAUSED **')
    if extra_lines:
      lines.extend(extra_lines)
    for role in sorted(role_counts.keys()):
      lines.append(f'  {role}: {role_counts[role]}')
    y = 6
    for line in lines:
      surf = self._font.render(line, True, (230, 230, 230))
      self._screen.blit(surf, (8, y))
      y += 16

  def _redraw(self, step_idx, extra_lines=None):
    self._draw_environment()
    self._draw_pheromones(self.sim.env)
    self._draw_agents()
    self._draw_hud(step_idx, extra_lines=extra_lines)
    pygame.display.flip()

  def render_frame(self, simulation, step_idx):
    """Callback for Simulation.run(); return False to stop early."""
    self._init_pygame()
    self._last_step_idx = step_idx
    if self.total_steps and step_idx >= self.total_steps - 1:
      self.finished = True
    if not self._handle_events():
      return False

    if self.paused:
      self._redraw(step_idx)
      self._clock.tick(self.cfg['fps'])
      return True

    self._redraw(step_idx)
    self._clock.tick(int(self.cfg['fps'] * self.speed_mult))
    return True

  def wait_for_close(self, message=None):
    """Poll events after simulation ends until user closes or auto-close."""
    if not self._initialized:
      return
    msg = message or 'Press Esc or close window to exit'
    auto_close = self.cfg.get('auto_close_seconds', 0)
    start = time.time()
    self.finished = True
    while True:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
          return
      if auto_close and (time.time() - start) >= auto_close:
        return
      self._redraw(self._last_step_idx, extra_lines=[msg])
      self._clock.tick(30)

  def close(self):
    if self._initialized:
      pygame.quit()
      self._initialized = False
