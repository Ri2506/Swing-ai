# SwingAI - Start Here

This repo contains a full-stack swing trading platform for Indian markets, plus ML feature/strategy code and model inference endpoints.

## 1) High-level components
- Backend API: `src/backend/api/app.py` (FastAPI)
- Frontend UI: `src/frontend` (Next.js 14)
- ML/AI core: `ml/` (features, filters, strategies, ensemble, inference)
- Infrastructure: `infrastructure/database/complete_schema.sql`

## 2) How to run locally

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn src.backend.api.app:app --reload --port 8000
```

In a new terminal:

```bash
cd src/frontend
npm install
npm run dev
```

Or run both with:

```bash
./scripts/dev.sh
```

## 3) What is implemented vs placeholders
- Signal generation (`src/backend/services/signal_generator.py`) uses PKScreener candidates and simulated feature values when no live feed is available; optional ML inference can be called via `ML_INFERENCE_URL`.
- Scheduler and realtime services are wired behind `ENABLE_SCHEDULER` and `ENABLE_REDIS` flags (Redis is optional; in-memory mode is supported).
- Trade execution defaults to DB-level updates; broker integrations exist under `src/backend/services/broker_integration.py` but are not fully wired for live execution.
- WebSocket endpoint is `/ws/{token}` (Supabase JWT). It currently responds to `ping`; channel subscriptions are not implemented server-side.
- ML training artifacts are not committed; Modal inference expects model files to be uploaded to a Modal volume.

## 4) Where to look next
- API routes and request/response schemas: `src/backend/api/app.py` and `src/backend/schemas/`
- PKScreener routes: `src/backend/api/screener_routes.py`
- Core ML pipeline: `ml/features/`, `ml/filters/`, `ml/models/`, `ml/strategies/`
- Modal inference endpoints: `ml/inference/modal_inference.py` and `ml/inference/modal_inference_v2.py`
- Supabase schema: `infrastructure/database/complete_schema.sql`

## 5) Documentation map
- `docs/API_DOCUMENTATION.md`
- `docs/DEPLOYMENT_GUIDE.md`
- `docs/MODEL_DEPLOYMENT.md`
- `docs/ENHANCED_AI_CORE_V2.md`
- `docs/STRATEGIES_SYSTEM.md`
- `docs/PROJECT_ANALYSIS.md`
