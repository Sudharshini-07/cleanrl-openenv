from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from cleanrl.environment import DataCleaningEnv
from cleanrl.models import Action

app = FastAPI(title="CleanRL OpenEnv API")

env = None



class ActionInput(BaseModel):
    operation: str
    column: Optional[str] = None
    strategy: Optional[str] = None



@app.post("/reset")
def reset():
    global env

    # default task (validator just checks reset works)
    env = DataCleaningEnv(task_id="easy")

    obs = env.reset()

    return {
        "observation": obs.model_dump()
    }


@app.post("/step")
def step(action: ActionInput):
    global env

    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}

    act = Action(**action.dict())

    obs, reward, done, info = env.step(act)

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    global env

    if env is None:
        return {"error": "Environment not initialized."}

    return env.state().model_dump()
