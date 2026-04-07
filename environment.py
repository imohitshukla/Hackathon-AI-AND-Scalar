# environment.py
# dynamic warehouse robot - enhanced version with 5 unique mechanics
# procedural gen, fire hazards, energy, fog of war, conveyor belts

from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Tuple
import uuid, copy, random
from collections import deque

class Action(BaseModel):
    action: str   # move_up/down/left/right, pickup, dropoff

class Observation(BaseModel):
    grid: List[List[str]]          # what the agent can see (fog masked)
    robot_pos: List[int]
    carrying: bool
    packages_remaining: int
    packages_delivered: int
    step_count: int
    max_steps: int
    task_instruction: str
    done: bool
    reward: float
    score: float                   # adding explicit score field for Phase 2 validator
    energy: int                    # -1 means infinite (tiers without energy)
    max_energy: int
    fire_count: int                # how many fire cells on the grid
    cells_visited: int
    tier: str
    error: Optional[str] = None

# cell types
EMPTY    = "."
WALL     = "#"
ROBOT    = "R"
PKG      = "P"
DROP     = "D"
FIRE     = "F"
RECHARGE = "E"
FOG      = "?"
# conveyors
CONV_UP    = "^"
CONV_DOWN  = "v"
CONV_LEFT  = "<"
CONV_RIGHT = ">"

CONVEYOR_CELLS = {CONV_UP, CONV_DOWN, CONV_LEFT, CONV_RIGHT}
WALKABLE = {EMPTY, PKG, DROP, RECHARGE, CONV_UP, CONV_DOWN, CONV_LEFT, CONV_RIGHT}

DIRS = {
    "move_up":    (-1, 0),
    "move_down":  ( 1, 0),
    "move_left":  ( 0,-1),
    "move_right": ( 0, 1),
}

CONV_DIRS = {
    CONV_UP:    (-1, 0),
    CONV_DOWN:  ( 1, 0),
    CONV_LEFT:  ( 0,-1),
    CONV_RIGHT: ( 0, 1),
}


# --- BFS utility to check if a path exists ---

def bfs_reachable(grid, start, targets):
    """check if all targets are reachable from start, avoiding walls and fire"""
    rows, cols = len(grid), len(grid[0])
    visited = set()
    queue = deque([tuple(start)])
    visited.add(tuple(start))
    found = set()

    target_set = {tuple(t) for t in targets}

    while queue:
        r, c = queue.popleft()
        if (r, c) in target_set:
            found.add((r, c))
            if found == target_set:
                return True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                cell = grid[nr][nc]
                if cell != WALL:  # fire is dynamic, don't block on it
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return found == target_set


# --- Procedural grid generation ---

def generate_grid(rows, cols, wall_density=0.15, seed=None):
    """make a random grid with walls, guaranteed connectivity from (0,0)"""
    rng = random.Random(seed)
    grid = [[EMPTY]*cols for _ in range(rows)]

    # sprinkle walls but verify we don't break connectivity
    candidates = []
    for r in range(rows):
        for c in range(cols):
            if (r == 0 and c == 0) or (r == rows-1 and c == cols-1):
                continue
            candidates.append((r, c))

    rng.shuffle(candidates)
    target_walls = int(rows * cols * wall_density)

    placed = 0
    for r, c in candidates:
        if placed >= target_walls:
            break
        grid[r][c] = WALL
        # quick check: can we still reach a good portion of cells?
        reachable = _count_reachable(grid, [0, 0])
        total_non_wall = sum(1 for rr in range(rows) for cc in range(cols) if grid[rr][cc] != WALL)
        if reachable < total_non_wall * 0.8:
            grid[r][c] = EMPTY  # undo, this wall disconnects too much
        else:
            placed += 1

    return grid


def _count_reachable(grid, start):
    """count how many non-wall cells are reachable from start"""
    rows, cols = len(grid), len(grid[0])
    visited = set()
    queue = deque([tuple(start)])
    visited.add(tuple(start))
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] != WALL:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return len(visited)


def place_items_safely(grid, robot_start, n_packages, n_dropoffs, n_recharges=0,
                       n_conveyors=0, seed=None):
    """place packages, dropoffs, recharges, conveyors on empty cells.
    retries placement until BFS confirms everything is reachable."""
    rng = random.Random(seed)
    rows, cols = len(grid), len(grid[0])

    for attempt in range(100):  # more retries for bigger grids
        g = copy.deepcopy(grid)
        empties = []
        for r in range(rows):
            for c in range(cols):
                if g[r][c] == EMPTY and [r,c] != robot_start:
                    empties.append([r, c])

        rng.shuffle(empties)
        if len(empties) < n_packages + n_dropoffs + n_recharges + n_conveyors:
            continue

        idx = 0
        packages = []
        for _ in range(n_packages):
            pos = empties[idx]; idx += 1
            packages.append(pos)
            g[pos[0]][pos[1]] = PKG

        dropoffs = []
        for _ in range(n_dropoffs):
            pos = empties[idx]; idx += 1
            dropoffs.append(pos)
            g[pos[0]][pos[1]] = DROP

        recharges = []
        for _ in range(n_recharges):
            pos = empties[idx]; idx += 1
            recharges.append(pos)
            g[pos[0]][pos[1]] = RECHARGE

        conv_types = [CONV_UP, CONV_DOWN, CONV_LEFT, CONV_RIGHT]
        conveyors = []
        for _ in range(n_conveyors):
            pos = empties[idx]; idx += 1
            ctype = rng.choice(conv_types)
            conveyors.append((pos, ctype))
            g[pos[0]][pos[1]] = ctype

        # verify everything is reachable
        all_targets = packages + dropoffs + recharges
        if bfs_reachable(g, robot_start, all_targets):
            return g, packages, dropoffs, recharges, conveyors

    # fallback: return what we have (shouldn't happen often)
    return g, packages, dropoffs, recharges, conveyors


# --- tier configs ---

def tier_rookie():
    # 5x5, no walls, 1 package, dead simple
    g = [["."]*5 for _ in range(5)]
    return {
        "grid": g, "robot_start": [0,0],
        "packages": [[2,3]], "dropoffs": [[4,4]],
        "recharges": [], "conveyors": [],
        "max_steps": 25, "energy": -1,
        "fire_enabled": False, "fog_enabled": False,
        "fire_spread_interval": 0,
        "instruction": (
            "TIER 1 — Rookie: 5x5 grid, no hazards. Pick up the package (P) "
            "and deliver to drop-off (D). "
            "Actions: move_up, move_down, move_left, move_right, pickup, dropoff."
        ),
    }

def tier_navigator():
    g = [
        list("...#..."),
        list("...#..."),
        list("......."),
        list("##...##"),
        list("......."),
        list("...#..."),
        list("...#..."),
    ]
    return {
        "grid": g, "robot_start": [0,0],
        "packages": [[1,5]], "dropoffs": [[6,0]],
        "recharges": [], "conveyors": [],
        "max_steps": 35, "energy": -1,
        "fire_enabled": False, "fog_enabled": False,
        "fire_spread_interval": 0,
        "instruction": (
            "TIER 2 — Navigator: 7x7 grid with walls (#). "
            "Route around barriers to deliver the package. "
            "Actions: move_up, move_down, move_left, move_right, pickup, dropoff."
        ),
    }

def tier_hauler():
    g = [
        list("....#..."),
        list(".#..#..."),
        list(".#....#."),
        list("...#..#."),
        list("...#...."),
        list(".#...#.."),
        list(".#...#.."),
        list("........"),
    ]
    return {
        "grid": g, "robot_start": [0,0],
        "packages": [[0,7],[3,0],[7,6]], "dropoffs": [[7,0],[0,5],[4,7]],
        "recharges": [], "conveyors": [],
        "max_steps": 50, "energy": -1,
        "fire_enabled": False, "fog_enabled": False,
        "fire_spread_interval": 0,
        "instruction": (
            "TIER 3 — Hauler: 8x8 grid, 3 packages to deliver. "
            "Carry one at a time to any drop-off zone. "
            "Actions: move_up, move_down, move_left, move_right, pickup, dropoff."
        ),
    }

def tier_survivor(seed=None):
    # 8x8 procedural, fire + energy
    rng_seed = seed or random.randint(0, 99999)
    g = generate_grid(8, 8, wall_density=0.12, seed=rng_seed)
    g, pkgs, drops, recharges, convs = place_items_safely(
        g, [0,0], n_packages=2, n_dropoffs=2, n_recharges=2,
        n_conveyors=0, seed=rng_seed
    )
    return {
        "grid": g, "robot_start": [0,0],
        "packages": pkgs, "dropoffs": drops,
        "recharges": [r for r in recharges], "conveyors": convs,
        "max_steps": 45, "energy": 35,
        "fire_enabled": True, "fog_enabled": False,
        "fire_spread_interval": 4,  # fire spreads every 4 steps
        "instruction": (
            "TIER 4 — Survivor: 8x8 procedural grid with FIRE (F) that spreads! "
            "You have limited ENERGY (moves cost 1, recharge at E stations). "
            "Avoid fire cells, manage energy, deliver 2 packages. "
            "Actions: move_up, move_down, move_left, move_right, pickup, dropoff."
        ),
    }

def tier_ghost_runner(seed=None):
    # 10x10 procedural, everything enabled
    rng_seed = seed or random.randint(0, 99999)
    g = generate_grid(10, 10, wall_density=0.10, seed=rng_seed)
    g, pkgs, drops, recharges, convs = place_items_safely(
        g, [0,0], n_packages=3, n_dropoffs=3, n_recharges=3,
        n_conveyors=4, seed=rng_seed
    )
    return {
        "grid": g, "robot_start": [0,0],
        "packages": pkgs, "dropoffs": drops,
        "recharges": [r for r in recharges], "conveyors": convs,
        "max_steps": 60, "energy": 50,
        "fire_enabled": True, "fog_enabled": True,
        "fire_spread_interval": 5,
        "instruction": (
            "TIER 5 — Ghost Runner: 10x10 procedural grid. FOG OF WAR limits "
            "visibility to 2 cells! Fire spreads, energy is limited, conveyor "
            "belts (^v<>) push you around. Deliver 3 packages to survive. "
            "Actions: move_up, move_down, move_left, move_right, pickup, dropoff."
        ),
    }

TIERS = {
    "tier1_rookie":       tier_rookie,
    "tier2_navigator":    tier_navigator,
    "tier3_hauler":       tier_hauler,
    "tier4_survivor":     tier_survivor,
    "tier5_ghost_runner": tier_ghost_runner,
}


class WarehouseEnv:
    """
    enhanced warehouse env with 5-tier ladder progression.
    each tier introduces new mechanics on top of the last.
    """
    def __init__(self):
        self.episode_id = ""
        self.step_count = 0
        self.max_steps = 0
        self.current_task = ""
        self.is_done = False
        self.reward = 0.01
        self.last_error = None

        self.grid = []
        self.nrows = 0
        self.ncols = 0
        self.robot_pos = [0, 0]
        self.carrying = False

        self.package_cells = []
        self.dropoff_cells = []
        self.recharge_cells = []
        self.conveyor_cells = []  # list of (pos, type)
        self.packages_delivered = 0
        self.total_packages = 0
        self.instruction = ""

        # new mechanics
        self.energy = -1       # -1 = infinite
        self.max_energy = -1
        self.fire_enabled = False
        self.fire_cells = []
        self.fire_spread_interval = 0
        self.fog_enabled = False
        self.visited_cells = set()
        self.fire_hits = 0
        self.total_visitable = 0

    def _apply_fog(self, grid):
        """mask everything outside 2-cell radius with '?'"""
        if not self.fog_enabled:
            return grid
        masked = [["?" for _ in row] for row in grid]
        rr, rc = self.robot_pos
        for r in range(self.nrows):
            for c in range(self.ncols):
                if abs(r - rr) + abs(c - rc) <= 2:
                    masked[r][c] = grid[r][c]
        return masked

    def _build_view(self):
        g = copy.deepcopy(self.grid)
        # put fire on the grid for display
        for fr, fc in self.fire_cells:
            if g[fr][fc] not in (WALL,):
                g[fr][fc] = FIRE
        r, c = self.robot_pos
        if g[r][c] in (EMPTY, RECHARGE) or g[r][c] in CONVEYOR_CELLS:
            g[r][c] = ROBOT
        return self._apply_fog(g)

    def _make_obs(self):
        return Observation(
            grid=self._build_view(),
            robot_pos=list(self.robot_pos),
            carrying=self.carrying,
            packages_remaining=len(self.package_cells),
            packages_delivered=self.packages_delivered,
            step_count=self.step_count,
            max_steps=self.max_steps,
            task_instruction=self.instruction,
            done=self.is_done,
            reward=float(min(0.999, max(0.001, self.reward))),
            score=float(min(0.999, max(0.001, self.reward))),
            energy=self.energy,
            max_energy=self.max_energy,
            fire_count=len(self.fire_cells),
            cells_visited=len(self.visited_cells),
            tier=self.current_task,
            error=self.last_error,
        )

    def reset(self, task_name: str) -> Observation:
        if task_name not in TIERS:
            raise ValueError(f"unknown tier '{task_name}', pick from {list(TIERS.keys())}")

        cfg = TIERS[task_name]()

        self.episode_id = str(uuid.uuid4())
        self.current_task = task_name
        self.step_count = 0
        self.is_done = False
        self.reward = 0.01
        self.last_error = None
        self.carrying = False
        self.packages_delivered = 0
        self.fire_hits = 0

        self.grid = copy.deepcopy(cfg["grid"])
        self.nrows = len(self.grid)
        self.ncols = len(self.grid[0])
        self.robot_pos = list(cfg["robot_start"])
        self.max_steps = cfg["max_steps"]
        self.instruction = cfg["instruction"]

        self.energy = cfg["energy"]
        self.max_energy = cfg["energy"]
        self.fire_enabled = cfg["fire_enabled"]
        self.fire_spread_interval = cfg["fire_spread_interval"]
        self.fog_enabled = cfg["fog_enabled"]

        self.package_cells = [list(p) for p in cfg["packages"]]
        self.dropoff_cells = [list(d) for d in cfg["dropoffs"]]
        self.recharge_cells = [list(r) for r in cfg.get("recharges", [])]
        self.conveyor_cells = [(list(pos), ctype) for pos, ctype in cfg.get("conveyors", [])]
        self.total_packages = len(self.package_cells)

        self.fire_cells = []
        self.visited_cells = {tuple(self.robot_pos)}

        # count visitable cells for exploration score
        self.total_visitable = sum(
            1 for r in range(self.nrows) for c in range(self.ncols)
            if self.grid[r][c] != WALL
        )

        # stamp items onto grid
        for pr, pc in self.package_cells:
            self.grid[pr][pc] = PKG
        for dr, dc in self.dropoff_cells:
            if self.grid[dr][dc] == EMPTY:
                self.grid[dr][dc] = DROP
        for rr, rc in self.recharge_cells:
            if self.grid[rr][rc] == EMPTY:
                self.grid[rr][rc] = RECHARGE
        for (cr, cc), ctype in self.conveyor_cells:
            if self.grid[cr][cc] == EMPTY:
                self.grid[cr][cc] = ctype

        # seed initial fire in tiers that use it
        if self.fire_enabled:
            rng = random.Random(self.episode_id)
            empties = [
                [r,c] for r in range(self.nrows) for c in range(self.ncols)
                if self.grid[r][c] == EMPTY and [r,c] != self.robot_pos
            ]
            if empties:
                fire_start = rng.choice(empties)
                self.fire_cells.append(fire_start)

        return self._make_obs()

    def _spread_fire(self):
        """expand fire to adjacent empty cells"""
        if not self.fire_enabled or not self.fire_cells:
            return
        if self.fire_spread_interval <= 0:
            return
        if self.step_count % self.fire_spread_interval != 0:
            return

        new_fires = []
        for fr, fc in self.fire_cells:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = fr+dr, fc+dc
                if 0 <= nr < self.nrows and 0 <= nc < self.ncols:
                    if self.grid[nr][nc] == EMPTY and [nr,nc] not in self.fire_cells and [nr,nc] not in new_fires:
                        new_fires.append([nr, nc])
        # only spread to 1-2 cells max per tick so it doesn't explode
        rng = random.Random(self.step_count)
        rng.shuffle(new_fires)
        self.fire_cells.extend(new_fires[:2])

    def _check_conveyor(self):
        """if robot is on a conveyor, push it"""
        r, c = self.robot_pos
        cell = self.grid[r][c]
        if cell in CONV_DIRS:
            dr, dc = CONV_DIRS[cell]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.nrows and 0 <= nc < self.ncols:
                if self.grid[nr][nc] != WALL and [nr, nc] not in self.fire_cells:
                    self.robot_pos = [nr, nc]

    def step(self, action: Action) -> Tuple[Observation, float, bool, Optional[str]]:
        self.last_error = None

        if self.is_done:
            self.last_error = "Episode already done."
            return self._make_obs(), float(min(0.999, max(0.001, self.reward))), True, self.last_error

        self.step_count += 1
        cmd = action.action.strip().lower()
        r, c = self.robot_pos

        # energy check
        if self.energy == 0:
            self.is_done = True
            self.last_error = "Out of energy!"
            self.reward = self._compute_reward()
            return self._make_obs(), float(min(0.999, max(0.001, self.reward))), True, self.last_error

        if cmd in DIRS:
            dr, dc = DIRS[cmd]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.nrows and 0 <= nc < self.ncols:
                target = self.grid[nr][nc]
                if target == WALL:
                    self.last_error = "Blocked: wall or boundary."
                elif [nr, nc] in self.fire_cells:
                    # can walk through fire but take a hit
                    self.robot_pos = [nr, nc]
                    self.fire_hits += 1
                    self.last_error = "Ouch! Walked into fire."
                    if self.energy > 0:
                        self.energy -= 2  # fire costs extra energy
                else:
                    self.robot_pos = [nr, nc]
                    if self.energy > 0:
                        self.energy -= 1
            else:
                self.last_error = "Blocked: wall or boundary."

        elif cmd == "pickup":
            if self.carrying:
                self.last_error = "Already carrying a package."
            elif self.robot_pos in self.package_cells:
                self.carrying = True
                self.package_cells.remove(self.robot_pos)
                pr, pc = self.robot_pos
                self.grid[pr][pc] = EMPTY
            else:
                self.last_error = "No package here to pick up."

        elif cmd == "dropoff":
            if not self.carrying:
                self.last_error = "Not carrying a package."
            elif self.robot_pos in self.dropoff_cells:
                self.carrying = False
                self.packages_delivered += 1
                self.dropoff_cells.remove(self.robot_pos)
                dr2, dc2 = self.robot_pos
                self.grid[dr2][dc2] = EMPTY
            else:
                self.last_error = "Not at a drop-off zone."
        else:
            self.last_error = f"Unknown action '{cmd}'."

        # recharge check
        if self.robot_pos in self.recharge_cells and self.energy >= 0:
            self.energy = min(self.max_energy, self.energy + 10)
            self.recharge_cells.remove(self.robot_pos)
            rr2, rc2 = self.robot_pos
            self.grid[rr2][rc2] = EMPTY
            # no error to report, just silently recharge

        # track visited cells
        self.visited_cells.add(tuple(self.robot_pos))

        # conveyor push
        self._check_conveyor()
        self.visited_cells.add(tuple(self.robot_pos))

        # fire spread
        self._spread_fire()

        # clamp energy
        if self.energy != -1:
            self.energy = max(0, self.energy)

        # compute reward
        self.reward = self._compute_reward()

        # done conditions
        if self.packages_delivered == self.total_packages:
            self.is_done = True
        elif self.step_count >= self.max_steps:
            self.is_done = True
            if not self.last_error:
                self.last_error = "Max steps reached."
        elif self.energy == 0:
            self.is_done = True
            if not self.last_error:
                self.last_error = "Out of energy!"

        return self._make_obs(), float(min(0.999, max(0.001, self.reward))), self.is_done, self.last_error

    def state(self) -> Dict[str, Any]:
        val = float(min(0.999, max(0.001, self.reward)))
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "task": self.current_task,
            "reward": val,
            "score": val,
            "done": self.is_done,
        }

    # --- composite reward ---

    def _compute_reward(self):
        task = self.current_task

        delivery = self.packages_delivered / max(1, self.total_packages)
        efficiency = max(0, (self.max_steps - self.step_count)) / self.max_steps

        # for tiers 1-3, keep it simple
        if task in ("tier1_rookie", "tier2_navigator"):
            if self.packages_delivered >= self.total_packages:
                score = 0.85 + efficiency * 0.15
            elif self.carrying:
                score = 0.4
            else:
                score = 0.0

        elif task == "tier3_hauler":
            score = delivery * 0.85
            if self.packages_delivered == self.total_packages:
                score += efficiency * 0.15

        else:
            # tiers 4 and 5: composite of 4 signals
            max_fire_hits = 5
            safety = max(0.0, 1.0 - self.fire_hits / max_fire_hits)
            exploration = len(self.visited_cells) / max(1, self.total_visitable)

            score = (
                delivery    * 0.50 +
                efficiency  * 0.20 +
                safety      * 0.20 +
                exploration * 0.10
            )

            # bonus if fully delivered
            if self.packages_delivered == self.total_packages:
                score = max(score, 0.85)  # minimum 0.85 for full delivery

        return float(min(0.999, max(0.001, score)))
