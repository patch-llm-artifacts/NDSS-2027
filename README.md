This repository contains the anonymized research artifacts required to evaluate the contributions of the submitted paper. The artifacts support the construction of code-evolution provenance graphs (CEPGs), training and fine-tuning of a temporal graph neural network (TGAT), and evaluation of a reinforcement-learning–based patch generation pipeline.

This artifact uses the follwing datasets. Due to licensing constraints, raw datasets are not redistributed in this repository.

### Project CodeNet
We build Code-Evolution Provenance Graphs (CEPGs) from the Project CodeNet distribution using the dataset-provided CSV metadata and source files.
Download and dataset tools: https://github.com/IBM/Project_CodeNet

### Juliet
We build Juliet CEPGs from a hosted Juliet Test Suite dataset available through Hugging Face.
Dataset page: https://huggingface.co/datasets/LorenzH/juliet_test_suite_c_1_3

### ManyBugs
We build ManyBugs CEPGs from the ManyBugs benchmark.
Official benchmark page: https://repairbenchmarks.cs.umass.edu/

### CVEFixes
We use the CVEFixes dataset for evaluating the generalizability of the RL patch generation pipeline on real-world vulnerabilities.
Dataset page: https://github.com/secureIT-project/CVEfixes


## Included Sample for Patch Environment Testing

To enable lightweight testing and validation of the patching pipeline without requiring full dataset downloads, this repository includes a small curated sample of vulnerable code snippets constructed from the Juliet and ManyBugs datasets.

- File: `proper_vulnerable_code.txt`
- Usage: automatically loaded by the patch environment when running in training mode

This sample allows verification of core functionality including compilation, action selection, reward computation, and patch application with minimal setup. It is intended for sanity checking and smoke testing only; full experimental results reported in the paper require the complete datasets.

## Reviewer Smoke Test Guide

The `smoke_test.py` script allows you to validate the project setup and environment without loading large datasets. It validates core imports, environment initialization, and runs a mini reinforcement learning training trial.

### Local Environment Setup

If you prefer to run the code locally rather than via Docker, follow these steps:

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```
2. **Activate the environment**:
   - Windows: `.\venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


### Running the Smoke Test

Execute the script using the Python interpreter in your environment:

```bash
python smoke_test.py
```

The test is successful if it concludes with `ALL SMOKE TESTS PASSED!`.

---

The experimental pipeline consists of the following stages:

1. **CodeNet CEPG Construction**
   - Build CEPGs from the Project CodeNet dataset using the metadata and source code.
   - Extract graph, node, and temporal features required for TGAT training.

2. **TGAT Pretraining on CodeNet**
   - Train a temporal graph attention network (TGAT) on CodeNet CEPGs to learn generic code-evolution representations.

3. **Juliet CEPG Construction and Fine-Tuning**
   - Build CEPGs from the Juliet vulnerability dataset.
   - Fine-tune the pretrained TGAT model on Juliet for vulnerability-aware representations.

4. **ManyBugs CEPG Construction and Evaluation**
   - Build CEPGs from the ManyBugs dataset.
   - Evaluate the Juliet-fine-tuned TGAT model on ManyBugs without further fine-tuning.

5. **Patch Generation with Reinforcement Learning**
   - Fine-tune a CodeT5 model on Juliet for patch generation.
   - Train a PPO-based reinforcement learning agent in a patching environment.
   - Evaluate patch success, exploit blocking, and patch quality.

6. **Comparative Evaluation on CVEFixes**
   - Evaluate the trained RL agent against modernbaselines (ChatRepair, SAN2PATCH) on real-world CVEs.

---

## Repository Structure

```text
src/
  Dataset_builders/        CEPG construction scripts (CodeNet, Juliet, ManyBugs)
  Train_scripts/           TGAT and CodeT5 training / fine-tuning scripts
  Patch_code/              RL patch environment and PPO agent
  Evaluation_scripts/      Evaluation and metric computation scripts
  baselines/               Comparative evaluation against baselines

README.md                  Instructions for obtaining external datasets and running the code
requirements.txt           Python dependencies
Dockerfile                 Dockerfile for full reproducibility
```

## Execution Environment and File Paths

All experiments were executed inside a Docker container. Within the container, the repository root is mounted at `/app`, and all file paths in the code are defined relative to this location. When running the provided Docker image, placing the required datasets and pretrained models in the documented locations under `/app` is sufficient to reproduce the experiments.

## How to Run the Code

### Build the Docker Image

From the repository root:

```bash
docker build -t patch-llm-artifact .
```

---

```bash
docker run --rm -it \
  -v $(pwd):/app \
  patch-llm-artifact \
  python src/Patch_code/train_patch_agent.py
```
Download each dataset following the official instructions and place it in the corresponding directory.

---

### CEPG Construction

```bash
python src/Dataset_builders/extract_subset.py
python src/Dataset_builders/build_cepg.py
```

```bash
python src/Dataset_builders/build_juliet_edges.py
```

```bash
python src/Dataset_builders/build_many_bugs.py
```

### TGAT Training and Evaluation


```bash
python src/Train_scripts/train_cepg_temporal.py
```

```bash
python src/Train_scripts/fine_tune_juliet.py
```

```bash
python src/Train_scripts/tgat_classifier.py
```

---

### CodeT5 Fine-Tuning (Required for Full Patch Generation)

```bash
python src/Train_scripts/CodeT5.py
```

---

### Patch Generation with Reinforcement Learning (Full Pipeline)

```bash
python src/Patch_code/train_patch_agent.py
```

---

## Notes on Artifact Availability

All artifacts required for evaluation are included or described above. Following conditional paper acceptance, non-anonymized links to the same artifacts will be provided for camera-ready submission.