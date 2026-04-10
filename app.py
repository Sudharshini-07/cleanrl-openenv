import sys
import os
import math

sys.path.append(os.getcwd())

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

from cleanrl.environment import DataCleaningEnv
from cleanrl.models import Action

app = FastAPI(title="CleanRL OpenEnv API")

env = None


# =========================
# JSON SAFE CLEANER
# =========================
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    return obj


# =========================
# INPUT MODEL
# =========================
class ActionInput(BaseModel):
    operation: str
    column: Optional[str] = None
    strategy: Optional[str] = None


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"status": "running"}


# =========================
# RESET
# =========================
@app.post("/reset")
def reset():
    global env
    env = DataCleaningEnv(task_id="easy")
    obs = env.reset()
    return clean_for_json(obs.model_dump())


# =========================
# STEP
# =========================
@app.post("/step")
def step(action: ActionInput):
    global env

    if env is None:
        return {
            "observation": {},
            "reward": 0.0,
            "done": True,
            "info": {"error": "Call /reset first"}
        }

    try:
        # ✅ FIXED HERE (IMPORTANT)
        act = Action(**action.model_dump())

        obs, reward, done, info = env.step(act)

        return {
            "observation": clean_for_json(obs.model_dump()),
            "reward": float(reward),
            "done": bool(done),
            "info": info if info else {}
        }

    except Exception as e:
        return {
            "observation": {},
            "reward": 0.0,
            "done": True,
            "info": {"error": str(e)}
        }


# =========================
# STATE
# =========================
@app.get("/state")
def state():
    global env

    if env is None:
        return {}

    return clean_for_json(env.state().model_dump())


# =========================
# MAIN FUNCTION (REQUIRED)
# =========================
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


# =========================
# ENTRY POINT (REQUIRED)
# =========================
if __name__ == "__main__":
    main()
