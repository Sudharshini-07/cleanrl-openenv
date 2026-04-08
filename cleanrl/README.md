---
title: CleanRL - Data Cleaning RL Environment
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - data-cleaning
---

# CleanRL — Data Cleaning RL Environment

> OpenEnv-compliant RL environment for training agents to clean real-world tabular datasets.

---

## Overview

Data scientists spend **60–80% of their time cleaning data**, making it the biggest bottleneck in ML pipelines. CleanRL provides a **structured RL environment** where agents learn to fix real-world data issues step-by-step using reward feedback.

This project focuses on **environment design + evaluation**, not just model performance.

---

## Key Features

- Real-world data cleaning tasks  
- Dense reward shaping (not just final score)  
- Deterministic baseline agent  
- OpenAI-compatible inference pipeline  
- Fully OpenEnv-compliant  

---

## Action Space

| Operation | Params | Description |
|---|---|---|
| fill_null | column, strategy | Fill missing values |
| drop_duplicates | — | Remove duplicates |
| fix_dtype | column, strategy | Convert data types |
| remove_outliers | column, strategy | Handle outliers |
| normalize_format | column, strategy | Standardize text |
| done | — | End task |

---

## Observation Space

At each step, the agent receives:

- Dataset shape and columns  
- Null counts  
- Duplicate count  
- Data types  
- Outlier counts  
- Sample rows  
- Feedback from last action  

---

## Tasks

| Task | Rows | Errors | Difficulty |
|------|------|--------|------------|
| Easy | 105 | 5 (known) | Easy |
| Medium | 520 | 6 (hidden) | Medium |
| Hard | 315 | 7 (hidden) | Hard |

---

## Reward Function

- +1/n per correct fix  
- -0.01 per step  
- penalty for invalid actions  
- final reward = score (0–1)  

---

## Baseline Performance

Deterministic rule-based agent:

| Task | Score |
|------|------|
| Easy | 0.80 |
| Medium | 0.83 |
| Hard | 1.00 |

**Average: 0.878**

---

## Inference Design

The inference pipeline follows an **OpenAI-compatible structure**, using environment variables:

- API_BASE_URL  
- MODEL_NAME  
- HF_TOKEN  

A rule-based agent is used as baseline to ensure:

- reproducibility  
- no external API dependency  
- stable evaluation  

The system supports plugging in LLM/RL agents easily.

---

## Setup

```bash
git clone https://huggingface.co/spaces/Sudharshini07/cleanrl
cd cleanrl
pip install -r requirements.txt

python app.py
python inference.py
