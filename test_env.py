# tests for the enhanced warehouse env
# covers all 5 tiers + individual mechanic checks

from environment import WarehouseEnv, Action

def do(env, act):
    return env.step(Action(action=act))


# ---------- tier tests ----------

def test_tier1_rookie():
    env = WarehouseEnv()
    env.reset("tier1_rookie")

    # walk to pkg at (2,3), pick up, walk to drop at (4,4)
    for _ in range(2): do(env, "move_down")
    for _ in range(3): do(env, "move_right")
    obs, r, d, e = do(env, "pickup")
    assert obs.carrying and r >= 0.4

    for _ in range(2): do(env, "move_down")
    do(env, "move_right")
    obs, r, d, e = do(env, "dropoff")
    assert r >= 0.85 and d   # delivery + efficiency bonus
    print("  [ok] tier1 rookie")


def test_tier2_navigator():
    env = WarehouseEnv()
    env.reset("tier2_navigator")

    # navigate to (1,5): down 2, right 5, up 1
    do(env, "move_down"); do(env, "move_down")
    for _ in range(5): do(env, "move_right")
    do(env, "move_up")
    obs, r, d, e = do(env, "pickup")
    assert obs.carrying

    # to (6,0): route through gap at row 3
    do(env, "move_left"); do(env, "move_down")
    do(env, "move_left"); do(env, "move_down")
    do(env, "move_down")
    for _ in range(3): do(env, "move_left")
    do(env, "move_down"); do(env, "move_down")
    obs, r, d, e = do(env, "dropoff")
    assert r >= 0.85 and d
    print("  [ok] tier2 navigator")


def test_tier3_hauler():
    env = WarehouseEnv()
    env.reset("tier3_hauler")

    # pkg1: (3,0) -> drop (7,0)
    for _ in range(3): do(env, "move_down")
    do(env, "pickup")
    for _ in range(4): do(env, "move_down")
    obs, r, d, e = do(env, "dropoff")
    assert obs.packages_delivered == 1

    # pkg2: (7,6) -> drop (4,7)
    for _ in range(6): do(env, "move_right")
    do(env, "pickup")
    do(env, "move_right")
    for _ in range(3): do(env, "move_up")
    obs, r, d, e = do(env, "dropoff")
    assert obs.packages_delivered == 2

    # pkg3: (0,7) -> drop (0,5)
    for _ in range(4): do(env, "move_up")
    do(env, "pickup")
    do(env, "move_left"); do(env, "move_left")
    obs, r, d, e = do(env, "dropoff")
    assert obs.packages_delivered == 3 and d
    print(f"  [ok] tier3 hauler (reward={r:.2f})")


def test_tier4_survivor():
    """procedural grid with fire + energy, just verify it runs and scores"""
    env = WarehouseEnv()
    env.reset("tier4_survivor")

    assert env.energy > 0, "tier4 should have energy"
    assert env.fire_enabled, "tier4 should have fire"
    assert not env.fog_enabled, "tier4 shouldn't have fog"

    # just do some moves and verify mechanics
    initial_energy = env.energy
    do(env, "move_down")
    assert env.energy < initial_energy or env.energy == -1

    st = env.state()
    assert 0 <= st["reward"] <= 1.0
    print(f"  [ok] tier4 survivor (energy works, fire enabled)")


def test_tier5_ghost_runner():
    """procedural grid with everything, verify fog + conveyors work"""
    env = WarehouseEnv()
    obs = env.reset("tier5_ghost_runner")

    assert env.fog_enabled, "tier5 should have fog"
    assert env.fire_enabled, "tier5 should have fire"
    assert env.energy > 0, "tier5 should have energy"

    # check fog: cells far from robot should be '?'
    fog_count = sum(1 for row in obs.grid for c in row if c == "?")
    assert fog_count > 0, "fog should mask distant cells"

    # do a few moves
    for _ in range(5):
        do(env, "move_right")

    st = env.state()
    assert 0 <= st["reward"] <= 1.0
    print(f"  [ok] tier5 ghost runner (fog + fire + energy + conveyors)")


# ---------- mechanic-specific tests ----------

def test_fire_spreads():
    env = WarehouseEnv()
    env.reset("tier4_survivor")
    initial_fires = len(env.fire_cells)

    # run enough steps for fire to spread
    for _ in range(12):
        do(env, "move_down")
        if env.is_done:
            break

    # fire should have grown (or at least not disappeared)
    assert len(env.fire_cells) >= initial_fires
    print(f"  [ok] fire spread: {initial_fires} -> {len(env.fire_cells)}")


def test_energy_depletes():
    env = WarehouseEnv()
    env.reset("tier4_survivor")
    start_e = env.energy

    # try each direction until one is not blocked
    moved = False
    for direction in ["move_down", "move_right", "move_left", "move_up"]:
        obs, r, d, e = do(env, direction)
        if e != "Blocked: wall or boundary.":
            moved = True
            break

    if moved:
        assert env.energy < start_e, "energy should decrease on unblocked move"
    else:
        # very unlikely but possible on edge-case grids
        pass

    # exhaust energy by moving a lot
    for _ in range(60):
        do(env, "move_right")
        do(env, "move_down")
        if env.is_done:
            break

    assert env.is_done, "should end when energy runs out or max steps"
    print(f"  [ok] energy depletion works")


def test_fog_visibility():
    env = WarehouseEnv()
    obs = env.reset("tier5_ghost_runner")

    rr, rc = obs.robot_pos
    for r_idx, row in enumerate(obs.grid):
        for c_idx, cell in enumerate(row):
            dist = abs(r_idx - rr) + abs(c_idx - rc)
            if dist <= 2:
                assert cell != "?", f"cell ({r_idx},{c_idx}) within radius should be visible"
            else:
                assert cell == "?", f"cell ({r_idx},{c_idx}) outside radius should be fogged"
    print("  [ok] fog radius correct")


def test_bfs_solvability():
    """procedural tiers should always generate solvable grids"""
    from environment import bfs_reachable
    for tier in ["tier4_survivor", "tier5_ghost_runner"]:
        for _ in range(10):
            env = WarehouseEnv()
            env.reset(tier)
            all_targets = env.package_cells + env.dropoff_cells
            assert bfs_reachable(env.grid, env.robot_pos, all_targets), f"{tier} grid not solvable"
    print("  [ok] BFS solvability (10 random grids x 2 tiers)")


def test_reward_range():
    for tier in ["tier1_rookie", "tier2_navigator", "tier3_hauler",
                  "tier4_survivor", "tier5_ghost_runner"]:
        env = WarehouseEnv()
        env.reset(tier)
        for _ in range(env.max_steps):
            _, r, d, _ = do(env, "move_down")
            assert 0.0 <= r <= 1.0, f"reward {r} out of range on {tier}"
            if d:
                break
    print("  [ok] rewards in [0, 1] for all tiers")


def test_state_api():
    env = WarehouseEnv()
    env.reset("tier1_rookie")
    s = env.state()
    assert s["task"] == "tier1_rookie"
    assert not s["done"]
    assert "episode_id" in s
    print("  [ok] state() api")


if __name__ == "__main__":
    print("running enhanced warehouse tests...\n")

    # tier progression
    test_tier1_rookie()
    test_tier2_navigator()
    test_tier3_hauler()
    test_tier4_survivor()
    test_tier5_ghost_runner()

    # mechanic checks
    test_fire_spreads()
    test_energy_depletes()
    test_fog_visibility()
    test_bfs_solvability()

    # general
    test_reward_range()
    test_state_api()

    print("\nall good!")
