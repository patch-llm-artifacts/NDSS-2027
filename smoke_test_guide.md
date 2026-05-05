# Patch LLM Artifact: Reviewer Smoke Test Guide

This document explains how to use `smoke_test.py` script to validate the project setup and environment before evaluation.

## Purpose

It provides a fast way to check if:
1. All core dependencies are installed.
2. The custom `PatchEnv` environment can be initialized and stepped.
3. A complete reinforcement learning training rollout and inference work correctly on the environment.


## How to Run

Execute the script using the Python interpreter in the virtual environment:

```bash
.\venv\Scripts\python.exe smoke_test.py
```

Or inside the project directory:

```bash
python smoke_test.py
```

## What the Smoke Test Validates

1. **Imports**: Validates key packages (`torch`, `gymnasium`, `stable_baselines3`, `torch_geometric`, `transformers`).
2. **Local Environment**: Sets up Python path correctly and tests loading the custom `PatchEnv` using curated vulnerable code samples.
3. **Environment Reset and Steps**: Tests that environment reset and stepping through all available actions returns the expected observations, rewards, and status flags without exceptions.
4. **Mini-Rollout**: Runs a PPO agent for 1 training iteration (`64` steps) on the `PatchEnv` and tests deterministic inference.


