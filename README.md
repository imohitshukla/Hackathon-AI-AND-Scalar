# Dynamic Warehouse Robot — OpenEnv

A 5-tier warehouse robot environment with progressively unlocking mechanics. Built for the OpenEnv framework.

## What makes it unique

Five mechanics layered into a ladder progression:

1. **Procedural generation** — tiers 4&5 generate random grids with BFS-verified solvability, so no two runs are alike
2. **Dynamic fire** — fire cells spread to adjacent squares every N steps. You can walk through fire but it costs extra energy and hurts your score
3. **Energy system** — limited fuel per episode. Each move costs 1 energy. Recharge at `E` stations or run out and die
4. **Fog of war** — in tier 5 you can only see 2 cells around you. Everything else is `?`. Forces exploration
5. **Conveyor belts** — `^v<>` cells auto-push your robot. Can be shortcuts or traps

## The ladder

| Tier | Name | Grid | Mechanics |
|------|------|------|-----------|
| 1 | Rookie | 5x5 | Just move, pickup, dropoff |
| 2 | Navigator | 7x7 | Walls to route around |
| 3 | Hauler | 8x8 | 3 packages, multi-delivery |
| 4 | Survivor | 8x8 procedural | Fire + Energy |
| 5 | Ghost Runner | 10x10 procedural | Fog + Conveyors + everything |

## Reward function

Tiers 1-3 use simple delivery + efficiency scoring. Tiers 4-5 use a composite:

```
reward = delivery(0.5) + efficiency(0.2) + safety(0.2) + exploration(0.1)
```

All rewards are 0.0-1.0.

## Setup

```bash
pip install -r requirements.txt

export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."

python inference.py
```

## Testing (no API key needed)

```bash
python test_env.py
```

Covers all 5 tiers, fire spreading, energy depletion, fog radius, BFS solvability, and reward ranges.

## Docker

```bash
docker build -t warehouse-robot-env .
docker run -e API_BASE_URL -e MODEL_NAME -e HF_TOKEN warehouse-robot-env
```

## Action space

`move_up`, `move_down`, `move_left`, `move_right`, `pickup`, `dropoff`

## Grid symbols

`. `empty  `#` wall  `P` package  `D` dropoff  `R` robot  `F` fire  `E` recharge  `?` fog  `^ v < >` conveyors
