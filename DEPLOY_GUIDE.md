# 🚀 SwingAI Solo Founder Deployment Guide

Complete step-by-step guide to deploy SwingAI using **Supabase**, **Railway**, **Vercel**, and **Modal**.

**Estimated Time:** 30-45 minutes  
**Monthly Cost:** ~$25-50 (can start free)

---

## 📋 Prerequisites

1. GitHub account with your code pushed
2. Credit card for paid tiers (optional, free tiers work)

---

## 🗄️ Step 1: Supabase (Database & Auth)

### 1.1 Create Project

1. Go to [supabase.com](https://supabase.com) → Sign up/Login
2. Click **"New Project"**
3. Fill in:
   - **Name:** `swingai-prod`
   - **Database Password:** (save this!)
   - **Region:** Mumbai (ap-south-1) for India
4. Click **"Create new project"** (wait 2-3 mins)

### 1.2 Run Database Schema

1. Go to **SQL Editor** (left sidebar)
2. Click **"New query"**
3. Copy entire contents of `infrastructure/database/complete_schema.sql`
4. Paste and click **"Run"**
5. You should see "Success. No rows returned"

### 1.3 Get API Keys

1. Go to **Settings** → **API**
2. Copy these values:
   - **Project URL** → `SUPABASE_URL`
   - **anon public** → `SUPABASE_ANON_KEY`
   - **service_role secret** → `SUPABASE_SERVICE_KEY`

### 1.4 Configure Auth

1. Go to **Authentication** → **Providers**
2. Enable **Email** (already enabled by default)
3. Optional: Enable **Google OAuth**
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create OAuth 2.0 credentials
   - Add redirect URL: `https://your-project.supabase.co/auth/v1/callback`

### 1.5 Insert Subscription Plans

Run this SQL to add plans:

```sql
INSERT INTO subscription_plans (name, display_name, price_monthly, price_quarterly, price_yearly, features, max_signals_per_day, max_positions, fo_enabled, sort_order) VALUES
('free', 'Free', 0, 0, 0, '{"signals": 3, "alerts": 5, "support": "community"}', 3, 2, false, 1),
('starter', 'Starter', 49900, 134900, 479900, '{"signals": 10, "alerts": 20, "support": "email"}', 10, 5, false, 2),
('pro', 'Pro', 149900, 404900, 1439900, '{"signals": "unlimited", "alerts": "unlimited", "support": "priority", "api": true}', -1, 15, true, 3),
('elite', 'Elite', 299900, 809900, 2879900, '{"signals": "unlimited", "alerts": "unlimited", "support": "dedicated", "api": true, "custom_scans": true}', -1, 30, true, 4);
```

---

## 🚂 Step 2: Railway (Backend)

### 2.1 Create Project

1. Go to [railway.app](https://railway.app) → Sign up with GitHub
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your `SwingAI` repository
5. Railway auto-detects Python

### 2.2 Configure Environment Variables

1. Click on your service → **Variables**
2. Add these (click "RAW Editor" for bulk paste):

```
APP_NAME=SwingAI
APP_VERSION=2.0.0
APP_ENV=production
DEBUG=false
SECRET_KEY=generate-a-32-char-random-string-here

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key

RAZORPAY_KEY_ID=rzp_live_xxxxx
RAZORPAY_KEY_SECRET=your-secret

FRONTEND_URL=https://your-app.vercel.app

RATE_LIMIT_PER_MINUTE=60
LOG_LEVEL=INFO
```

### 2.3 Configure Start Command

1. Go to **Settings** → **Deploy**
2. Set **Start Command**:
```bash
uvicorn src.backend.api.app:app --host 0.0.0.0 --port $PORT
```

### 2.4 Get Backend URL

1. Go to **Settings** → **Networking**
2. Click **"Generate Domain"**
3. Copy URL (e.g., `https://swingai-production.up.railway.app`)

---

## ▲ Step 3: Vercel (Frontend)

### 3.1 Create Project

1. Go to [vercel.com](https://vercel.com) → Sign up with GitHub
2. Click **"Add New"** → **"Project"**
3. Import your `SwingAI` repository
4. Configure:
   - **Framework Preset:** Next.js
   - **Root Directory:** `src/frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `.next`

### 3.2 Add Environment Variables

Add these in **Environment Variables**:

```
NEXT_PUBLIC_API_URL=https://swingai-production.up.railway.app
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
NEXT_PUBLIC_RAZORPAY_KEY_ID=rzp_live_xxxxx
```

### 3.3 Deploy

1. Click **"Deploy"**
2. Wait for build (2-3 mins)
3. Get your URL (e.g., `https://swingai.vercel.app`)

### 3.4 Update CORS

Go back to Railway and update:
```
ALLOWED_ORIGINS=https://swingai.vercel.app,https://www.swingai.vercel.app
FRONTEND_URL=https://swingai.vercel.app
```

---

## 🤖 Step 4: Modal (ML Inference)

### 4.1 Setup Modal

1. Go to [modal.com](https://modal.com) → Sign up
2. Install Modal CLI:
```bash
pip install modal
modal token new
```

### 4.2 Deploy Model

```bash
cd ml/inference
modal deploy modal_inference.py
```

### 4.3 Get Endpoint URL

After deploy, Modal shows:
```
https://your-username--swingai-inference-fastapi-app.modal.run
```

### 4.4 Update Railway

Add to Railway environment:
```
MODAL_INFERENCE_URL=https://your-username--swingai-inference-fastapi-app.modal.run
```

---

## 💳 Step 5: Razorpay (Payments)

### 5.1 Create Account

1. Go to [razorpay.com](https://razorpay.com) → Sign up
2. Complete KYC verification

### 5.2 Get API Keys

1. Go to **Settings** → **API Keys**
2. Generate keys:
   - **Test Mode** for development
   - **Live Mode** for production
3. Copy `Key ID` and `Key Secret`

### 5.3 Configure Webhook (Optional)

1. Go to **Settings** → **Webhooks**
2. Add endpoint: `https://your-backend.railway.app/api/webhooks/razorpay`
3. Select events: `payment.captured`, `subscription.activated`

---

## 🔧 Step 6: Final Configuration

### 6.1 Update Supabase Auth Redirect

1. Go to Supabase → **Authentication** → **URL Configuration**
2. Add to **Redirect URLs**:
   - `https://swingai.vercel.app/**`
   - `https://swingai.vercel.app/auth/callback`

### 6.2 Test Everything

1. **Health Check:**
```bash
curl https://your-backend.railway.app/api/health
```

2. **Frontend:** Visit your Vercel URL

3. **Create Test User:**
   - Sign up on your app
   - Check Supabase → Authentication → Users

4. **Test Payment (Test Mode):**
   - Use Razorpay test card: `4111 1111 1111 1111`

---

## 📊 Monitoring & Logs

### Railway Logs
```bash
railway logs
```
Or view in Railway dashboard → Deployments → Logs

### Vercel Logs
View in Vercel dashboard → Project → Deployments → Functions

### Supabase Logs
View in Supabase dashboard → Database → Logs

---

## 🔄 Continuous Deployment

Both Railway and Vercel auto-deploy on push to `main`:

```bash
git add .
git commit -m "feat: new feature"
git push origin main
```

---

## 💰 Cost Breakdown

| Service | Free Tier | Paid Tier |
|---------|-----------|-----------|
| Supabase | 500MB DB, 1GB storage | $25/mo |
| Railway | $5 free credit/mo | $5-20/mo |
| Vercel | 100GB bandwidth | $20/mo |
| Modal | $30 free credit | Pay per use |
| **Total** | **~$0-5/mo** | **~$25-50/mo** |

---

## 🚨 Troubleshooting

### Backend not starting
- Check Railway logs for errors
- Verify all env vars are set
- Ensure `requirements.txt` is complete

### Frontend build failing
- Check Vercel build logs
- Verify `package.json` dependencies
- Check for TypeScript errors

### Auth not working
- Verify Supabase URL and keys
- Check redirect URLs in Supabase
- Ensure CORS is configured

### Payments failing
- Use test mode keys first
- Check Razorpay dashboard for errors
- Verify webhook URL

---

## 🎉 You're Live!

Your SwingAI platform is now deployed:

- **Frontend:** `https://swingai.vercel.app`
- **Backend API:** `https://swingai-production.up.railway.app`
- **API Docs:** `https://swingai-production.up.railway.app/api/docs`

---

## 📱 Next Steps

1. **Custom Domain:** Add in Vercel/Railway settings
2. **SSL:** Automatic with Railway/Vercel
3. **Monitoring:** Add Sentry for error tracking
4. **Analytics:** Add Posthog or Mixpanel
5. **Backups:** Enable in Supabase dashboard

Need help? Create an issue on GitHub or check the docs!
