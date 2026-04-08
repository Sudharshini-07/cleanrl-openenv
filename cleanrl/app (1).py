"""
Gradio app for Hugging Face Spaces — interactive demo of CleanRL.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
from environment import DataCleaningEnv
from models import Action, CleaningOperation

# Global env instances
envs = {t: DataCleaningEnv(t) for t in ("easy", "medium", "hard")}
obs_store = {}

def reset_env(task_id):
    obs  = envs[task_id].reset()
    obs_store[task_id] = obs
    return format_obs(obs), "", ""

def step_env(task_id, action_json):
    try:
        action_dict = json.loads(action_json)
        action = Action(**action_dict)
    except Exception as e:
        return format_obs(obs_store.get(task_id)), f" Invalid JSON: {e}", ""

    obs, reward, done, info = envs[task_id].step(action)
    obs_store[task_id] = obs
    status = f"Reward: {reward:+.4f} | Done: {done}"
    if done:
        status += f" |  Final Score: {info.get('final_score', 0.0):.3f}"
    return format_obs(obs), status, json.dumps(envs[task_id].state(), indent=2)

def format_obs(obs) -> str:
    s = obs.dataset_stats
    lines = [
        f" Shape       : {s.shape}",
        f" Columns     : {s.columns}",
        f" Nulls       : {s.null_counts}",
        f" Duplicates  : {s.duplicate_count}",
        f" Dtypes      : {s.dtype_map}",
        f" Outliers    : {s.outlier_counts}",
        f"",
        f"Sample rows:",
    ] + [str(r) for r in s.sample_rows[:3]] + [
        f"",
        f" {obs.last_action_feedback}",
        f"Step {obs.step_count}/{obs.max_steps}",
    ]
    return "\n".join(lines)

EXAMPLE_ACTIONS = {
    "Fill nulls (mean)"        : '{"operation": "fill_null",        "column": "age",    "strategy": "mean"}',
    "Drop duplicates"          : '{"operation": "drop_duplicates"}',
    "Fix salary dtype"         : '{"operation": "fix_dtype",         "column": "salary", "strategy": "float"}',
    "Remove outliers (IQR)"    : '{"operation": "remove_outliers",   "column": "score",  "strategy": "iqr"}',
    "Normalize name (lower)"   : '{"operation": "normalize_format",  "column": "name",   "strategy": "lower"}',
    "Done"                     : '{"operation": "done"}',
}

with gr.Blocks(title="CleanRL — Data Cleaning RL Environment") as demo:
    gr.Markdown("#  CleanRL — Data Cleaning RL Environment\nTrain agents to clean messy real-world datasets.")

    with gr.Row():
        task_sel  = gr.Radio(["easy", "medium", "hard"], value="easy", label="Task difficulty")
        reset_btn = gr.Button(" Reset Environment", variant="primary")

    obs_box    = gr.Textbox(label=" Current Observation", lines=18, interactive=False)
    action_box = gr.Textbox(label=" Action JSON", placeholder='{"operation": "fill_null", "column": "age", "strategy": "mean"}')

    gr.Markdown("**Quick actions:**")
    with gr.Row():
        for label, val in EXAMPLE_ACTIONS.items():
            gr.Button(label, size="sm").click(lambda v=val: v, outputs=action_box)

    step_btn   = gr.Button(" Step", variant="primary")
    status_box = gr.Textbox(label=" Step result", interactive=False)
    state_box  = gr.Textbox(label=" Environment state (JSON)", lines=10, interactive=False)

    reset_btn.click(reset_env, inputs=task_sel, outputs=[obs_box, status_box, state_box])
    step_btn.click(step_env,  inputs=[task_sel, action_box], outputs=[obs_box, status_box, state_box])

    demo.load(reset_env, inputs=task_sel, outputs=[obs_box, status_box, state_box])

if __name__ == "__main__":
    demo.launch()
