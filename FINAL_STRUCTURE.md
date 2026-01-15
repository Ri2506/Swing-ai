# SwingAI - Current Structure

This is the current repository layout based on the code on disk.

## Directory tree

```
SwingAI/
├── src/
│   ├── backend/
│   │   ├── api/              # FastAPI app.py + screener routes
│   │   ├── core/             # config, database, security
│   │   ├── middleware/       # logging, security headers, rate limiter
│   │   ├── services/         # signals, risk, F&O, brokers, screener, scheduler
│   │   ├── models/
│   │   ├── schemas/
│   │   └── utils/
│   └── frontend/
│       ├── app/              # Next.js routes
│       ├── components/
│       ├── contexts/
│       ├── hooks/
│       ├── lib/
│       └── types/
├── ml/
│   ├── features/             # 70-feature pipeline
│   ├── filters/              # market regime + premium filters
│   ├── models/               # hierarchical ensemble
│   ├── strategies/           # 20 strategies + selector
│   ├── inference/            # Modal endpoints
│   └── notebooks/            # training notebook(s)
├── infrastructure/
│   └── database/             # Supabase schema
├── docs/                      # API, deployment, ML docs
├── scripts/                   # dev helper
├── requirements.txt
├── railway.toml
├── vercel.json
└── .env.example
```

## Entry points
- Backend: `src/backend/api/app.py`
- Frontend: `src/frontend/app/layout.tsx`
- ML inference: `ml/inference/modal_inference.py` and `ml/inference/modal_inference_v2.py`

## Notes
- `src/backend/api/main.py` is a legacy entrypoint and is not used by the default dev command.
- Training artifacts are not committed; `ml/notebooks/` contains the training notebook, and model files are expected to be uploaded to Modal.
