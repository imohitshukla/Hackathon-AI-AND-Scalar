# inference.py - runs the LLM agent through all 5 warehouse tiers
#
# env vars:
#   API_BASE_URL  - llm endpoint
#   MODEL_NAME    - model to use
#   HF_TOKEN      - api key

import os, json
from openai import OpenAI
from environment import WarehouseEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "take_action",
        "description": "Execute one action. Valid: move_up, move_down, move_left, move_right, pickup, dropoff.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["move_up", "move_down", "move_left", "move_right", "pickup", "dropoff"],
                    "description": "which action to take",
                }
            },
            "required": ["action"],
        },
    },
}]

SYS_PROMPT = """\
You are a warehouse robot navigating a 2D grid.

Grid cells:
  . = empty   # = wall   P = package   D = drop-off   R = you
  F = fire (avoid!)   E = recharge station   ? = fog (unexplored)
  ^ v < > = conveyor belts (push you in that direction)

RULES:
- Pick up packages with 'pickup', deliver with 'dropoff' at D cells
- Fire (F) spreads over time — plan ahead to avoid it
- Energy is limited in later tiers — recharge at E stations
- Fog (?) means you can only see 2 cells around you
- Conveyors auto-push you after you step on them

Call take_action with exactly ONE action per turn. Think about your path carefully.
"""


def grid_to_str(grid):
    return "\n".join(" ".join(row) for row in grid)


def run_episode(task, client):
    env = WarehouseEnv()
    obs = env.reset(task)

    print(f"[START] task={task} env=warehouse-robot-env model={MODEL_NAME}")

    # build the initial context
    extra = ""
    if obs.energy >= 0:
        extra += f"\nEnergy: {obs.energy}/{obs.max_energy}"
    if obs.fire_count > 0:
        extra += f"\nFire cells: {obs.fire_count}"

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": (
            f"{obs.task_instruction}\n\n"
            f"Grid ({len(obs.grid)}x{len(obs.grid[0])}):\n{grid_to_str(obs.grid)}\n"
            f"Position: {obs.robot_pos}  Carrying: {obs.carrying}\n"
            f"Packages remaining: {obs.packages_remaining}"
            f"{extra}"
        )},
    ]

    rewards_log = []

    while not env.is_done:
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="required",
            )
        except Exception as exc:
            rewards_log.append(env.reward)
            print(
                f"[STEP]  step={env.step_count + 1} "
                f'action={{"action":"api_error"}} '
                f"reward={env.reward} done=true error={exc}"
            )
            break

        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            noop = Action(action="noop")
            obs_out, rew, done, err = env.step(noop)
            rewards_log.append(rew)
            print(
                f"[STEP]  step={env.step_count} "
                f'action={{"action":"noop"}} '
                f"reward={rew} done={str(done).lower()} "
                f"error=Agent did not call a tool"
            )
            messages.append({"role": "user", "content": "You need to call take_action."})
            continue

        tc = msg.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {"action": "noop"}

        chosen = args.get("action", "noop")
        obs_out, rew, done, err = env.step(Action(action=chosen))
        rewards_log.append(rew)

        err_out = err if err else "null"
        print(
            f'[STEP]  step={env.step_count} action={{"action":"{chosen}"}} '
            f"reward={rew} done={str(done).lower()} error={err_out}"
        )

        # feedback for the model
        status_bits = (
            f"Grid:\n{grid_to_str(obs_out.grid)}\n"
            f"Pos: {obs_out.robot_pos}  Carrying: {obs_out.carrying}\n"
            f"Pkgs left: {obs_out.packages_remaining}  "
            f"Delivered: {obs_out.packages_delivered}"
        )
        if obs_out.energy >= 0:
            status_bits += f"\nEnergy: {obs_out.energy}/{obs_out.max_energy}"
        if obs_out.fire_count > 0:
            status_bits += f"\nFire cells: {obs_out.fire_count}"
        status_bits += f"\nReward: {obs_out.reward}  Error: {obs_out.error}"

        messages.append({"role": "tool", "tool_call_id": tc.id, "content": status_bits})

    ok = env.reward >= 0.85  # tiers give ~0.85+ for full delivery
    rstr = ",".join(f"{r}" for r in rewards_log) if rewards_log else "0.01"
    print(f"[END]   success={str(ok).lower()} steps={env.step_count} rewards={rstr}")


if __name__ == "__main__":
    client = OpenAI(
        api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "dummy-key"),
        base_url=API_BASE_URL,
    )

    for tier in ["tier1_rookie", "tier2_navigator", "tier3_hauler",
                  "tier4_survivor", "tier5_ghost_runner"]:
        run_episode(tier, client)
