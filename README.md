# 🚀 SwingAI - AI-Powered Swing Trading Platform

**Production-ready AI trading platform for Indian stock markets**

Built with **Supabase** + **Railway** + **Vercel** + **Modal** | Perfect for solo founders

---

## ✨ Features

- 🤖 **AI Ensemble**: CatBoost + TFT + Stockformer models
- 📊 **40+ PKScreener Scans**: Stage 2 breakouts, VCP patterns, momentum
- 💹 **F&O Support**: Futures & Options with Greeks calculation
- 🔒 **5-Layer Risk Management**: Signal quality → Position sizing → Portfolio limits
- 🔌 **Multi-Broker**: Zerodha, Angel One, Upstox integration
- 💳 **Razorpay Payments**: Subscription management built-in
- 📱 **Real-time WebSocket**: Live price updates & notifications

---

## ⚡ Quick Start (Local Development)

```bash
# 1. Clone and install
git clone https://github.com/yourusername/SwingAI.git
cd SwingAI
pip install -r requirements.txt

# 2. Set up environment (create .env file with your keys)
# Required: SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY

# 3. Run backend
uvicorn src.backend.api.app:app --reload --port 8000

# 4. Run frontend (new terminal)
cd src/frontend
npm install
npm run dev
```

**Frontend**: http://localhost:3000  
**Backend**: http://localhost:8000  
**API Docs**: http://localhost:8000/api/docs

---

## 🚀 Deploy to Production (30 mins)

### Step 1: Supabase (Database)
1. Create project at [supabase.com](https://supabase.com)
2. Run `infrastructure/database/complete_schema.sql` in SQL Editor
3. Copy API keys

### Step 2: Railway (Backend)
1. Connect GitHub repo at [railway.app](https://railway.app)
2. Add environment variables
3. Deploy automatically

### Step 3: Vercel (Frontend)
1. Import repo at [vercel.com](https://vercel.com)
2. Set root directory: `src/frontend`
3. Add environment variables
4. Deploy

### Step 4: Modal (AI Models)
```bash
pip install modal
modal token new
modal deploy ml/inference/modal_inference.py
```

📖 **Full guide**: [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)

---

## 📁 Project Structure

```
SwingAI/
├── src/
│   ├── backend/              # FastAPI Backend
│   │   ├── api/app.py       # Main API with all routes
│   │   ├── core/            # Config, Database, Security
│   │   ├── middleware/      # Rate limiting, Logging
│   │   ├── services/        # Business logic
│   │   │   ├── signal_generator.py    # AI signal generation
│   │   │   ├── risk_management.py     # 5-layer risk engine
│   │   │   ├── fo_trading_engine.py   # F&O calculations
│   │   │   └── broker_integration.py  # Multi-broker support
│   │   └── schemas/         # Pydantic models
│   │
│   └── frontend/            # Next.js 14 Frontend
│       ├── app/             # Pages (dashboard, signals, portfolio, etc.)
│       ├── components/      # 15+ dashboard components
│       ├── contexts/        # Auth context
│       └── lib/             # API client, Supabase
│
├── infrastructure/
│   └── database/complete_schema.sql  # Full Supabase schema
│
├── ml/
│   ├── inference/modal_inference.py  # Modal deployment
│   └── training/                     # Model training scripts
│
├── .github/workflows/deploy.yml      # CI/CD pipeline
└── DEPLOY_GUIDE.md                   # Step-by-step deployment
```

---

## 🔑 Environment Variables

```env
# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=xxx
SUPABASE_SERVICE_KEY=xxx

# Razorpay
RAZORPAY_KEY_ID=rzp_xxx
RAZORPAY_KEY_SECRET=xxx

# Frontend
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=xxx
```

---

## 💰 Cost Breakdown

| Service | Free Tier | Paid |
|---------|-----------|------|
| Supabase | 500MB DB | $25/mo |
| Railway | $5 credit | ~$10/mo |
| Vercel | 100GB BW | $0-20/mo |
| Modal | $30 credit | ~$10/mo |
| **Total** | **$0-5/mo** | **~$25-50/mo** |

---

## 📚 Documentation

- 📖 [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md) - Complete deployment guide
- 🏗️ [START_HERE.md](START_HERE.md) - Project overview
- 📡 [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) - API reference

---

## 🛠️ Tech Stack

**Backend**: Python 3.11, FastAPI, Supabase, Razorpay  
**Frontend**: Next.js 14, React, Tailwind CSS, Framer Motion  
**AI/ML**: CatBoost, PyTorch, Modal  
**Infrastructure**: Railway, Vercel, Supabase, GitHub Actions

---

## 📄 License

MIT License - Free to use for personal and commercial projects.

---

**Built with ❤️ for Indian Traders**
