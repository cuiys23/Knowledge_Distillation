# Copilot instructions for `Knowledge_Distillation`

## Build, test, and lint commands

This repository does not currently include committed build scripts, lint config, or automated test suites (no `pyproject.toml`, `requirements*.txt`, `pytest.ini`, `Makefile`, or test files were found).

Use the Hydra entrypoints for runnable checks and experiments:

```bash
# 3-stage pipeline: federated -> distillation -> federated
python main.py

# 6-stage streaming pipeline
python main_streaming.py
```

For a quick single-run smoke check, override rounds/epochs from the CLI:

```bash
python main.py num_rounds=1 num_epochs=1
```

## High-level architecture

- `main.py` and `main_streaming.py` are orchestration entrypoints. They deep-copy Hydra config and run multi-stage workflows with different flags per stage.
- Shared FL execution is in `src/core/federated_runner.py::run_federated`:
  - loads and partitions battery dataset (`src/data/dataset.py`, `src/data/dataset_preparation.py`)
  - constructs Flower clients (`src/fl/client.py`)
  - runs simulation with custom strategy (`src/fl/strategy.py`)
  - performs centralized evaluation and saves `model.pkl` from final round (`src/fl/server.py`)
  - writes metrics/confusion-matrix artifacts under stage output dirs.
- Distillation logic is in `src/core/knowledge_distillation.py::run_knowledge_distillation`, which loads a source model and trains a student (`SimpleViT`/`ModifiedSimpleViT`) with CE + feature-map MSE variants.
- Model definitions and training utilities are in `src/models/simple_vit.py` and `src/models/models.py`.

## Key repository conventions

- **Hydra output directories are the stage contract.** Each stage writes artifacts (especially `model.pkl`) into its stage folder, and later stages load from `save_path_source` / `save_path_distillation`.
- **Client `cid == 0` is special.** `run_federated` always loads `num_clients + 1` partitions first; when `add_client` is false, partition 0 is dropped. When enabled, client 0 can act as the communication-limited/virtual client and has custom behavior in `FlowerClient.fit`.
- **Strategy behavior is flag-driven.** `FedAvgWithKnowledge` changes aggregation based on `weight` and `dynamic_weight`, and may blend in distillation model weights from disk.
- **Distillation mode is string-selected.** `cfg.distillation` must match implemented branches (`"base_train"`, `"distillation_adjust"`, `"distillation"`).
- **`train.unique` injects extra class data into one client.** `load_datasets` concatenates `ignore_class` data into `datasets[train.unique_one]` when enabled.
- **Keep config mutation local to stage copies.** Existing entrypoints use `copy.deepcopy(cfg)` per stage and mutate only stage-specific copies (do not rewrite `conf/config.yaml` during runs).
