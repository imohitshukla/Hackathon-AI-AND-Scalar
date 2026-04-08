# app.py - HTTP API server for HF Spaces deployment
# exposes the warehouse env via REST so the validator can hit it

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from environment import WarehouseEnv, Action, Observation

app = FastAPI(title="Dynamic Warehouse Robot - OpenEnv")

# strict clamp: validator requires 0 < score < 1
def _clamp(v: float) -> float:
    return float(max(0.0001, min(0.9999, v)))

# keep one env instance per session (good enough for eval)
env = WarehouseEnv()


class ResetRequest(BaseModel):
    task_name: str = "tier1_rookie"

class StepRequest(BaseModel):
    action: str

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    score: float
    done: bool
    error: Optional[str] = None


@app.get("/")
def health():
    return {"status": "ok", "env": "warehouse-robot-env", "tiers": list(WarehouseEnv.__init__.__code__.co_varnames)[:5]}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    # if validator sends no body, default to tier1
    t_name = req.task_name if req else "tier1_rookie"
    obs = env.reset(t_name)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    action = Action(action=req.action)
    obs, reward, done, error = env.step(action)
    clamped = _clamp(reward)
    return StepResponse(
        observation=obs,
        reward=clamped,
        score=clamped,
        done=done,
        error=error,
    ).model_dump()


@app.get("/state")
def state():
    s = env.state()
    s["reward"] = _clamp(s.get("reward", 0.0001))
    s["score"] = _clamp(s.get("score", 0.0001))
    return s


@app.get("/tasks")
def list_tasks():
    from environment import TIERS
    return {"tasks": list(TIERS.keys())}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

