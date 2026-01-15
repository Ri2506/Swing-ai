# SwingAI Model Deployment Guide

This repo ships inference endpoints for Modal and expects model artifacts to be uploaded to Modal volumes. Training scripts are not included; a notebook lives in `ml/notebooks/`.

## Inference options

### v1 (CatBoost + simulated scores)
- Entrypoint: `ml/inference/modal_inference.py`
- Volume: `swingai-models`
- Expected files:
  - `catboost_model.cbm`
  - `model_config.json` (optional: list of feature columns)

### v2 (5-model ensemble)
- Entrypoint: `ml/inference/modal_inference_v2.py`
- Volume: `swingai-models-v2`
- Expected files:
  - `tft_model.pt`
  - `lstm_model.pt`
  - `xgboost_model.json`
  - `rf_model.pkl`
  - `svm_model.pkl`
  - `model_config_v2.json` (feature columns)

## Deploy to Modal

```bash
pip install modal
modal token new
```

### Deploy v1

```bash
modal deploy ml/inference/modal_inference.py
```

### Deploy v2

```bash
modal deploy ml/inference/modal_inference_v2.py
```

## Upload model artifacts to Modal volumes

Example (v1):

```bash
modal volume create swingai-models
modal volume put swingai-models catboost_model.cbm /catboost_model.cbm
modal volume put swingai-models model_config.json /model_config.json
```

Example (v2):

```bash
modal volume create swingai-models-v2
modal volume put swingai-models-v2 tft_model.pt /tft_model.pt
modal volume put swingai-models-v2 lstm_model.pt /lstm_model.pt
modal volume put swingai-models-v2 xgboost_model.json /xgboost_model.json
modal volume put swingai-models-v2 rf_model.pkl /rf_model.pkl
modal volume put swingai-models-v2 svm_model.pkl /svm_model.pkl
modal volume put swingai-models-v2 model_config_v2.json /model_config_v2.json
```

## Backend configuration

`SignalGenerator` uses `ML_INFERENCE_URL` for the standard (v1-style) inference path.
If you enable the enhanced AI core (`ENABLE_ENHANCED_AI=true`), set `ENHANCED_ML_INFERENCE_URL` to the v2 endpoint.

```
ML_INFERENCE_URL=https://<modal-endpoint>
ENHANCED_ML_INFERENCE_URL=https://<modal-v2-endpoint>
```

## Performance tracking

The backend exposes `/api/signals/performance`, backed by the `model_performance` table in Supabase (`infrastructure/database/complete_schema.sql`). Populate that table from your training or evaluation pipeline.
