# 🚀 SwingAI Production Deployment Guide

Complete step-by-step guide to deploy SwingAI to production.

---

## 📋 Prerequisites

- [ ] Supabase account
- [ ] Razorpay account (for payments)
- [ ] Railway/Render account (for backend)
- [ ] Vercel account (for frontend)
- [ ] Modal account (for ML models)
- [ ] Domain name (optional but recommended)

---

## 🗄️ Step 1: Database Setup (Supabase)

### 1.1 Create Project
1. Go to [supabase.com](https://supabase.com)
2. Click "New Project"
3. Choose region: **Singapore** (closest to India)
4. Set strong database password
5. Wait for project to be ready (~2 minutes)

### 1.2 Run Database Schema
1. Go to **SQL Editor** in Supabase dashboard
2. Copy entire contents of `database/complete_schema.sql`
3. Paste and run
4. Verify: Check **Table Editor** - should see 12 tables

### 1.3 Get API Keys
1. Go to **Settings** → **API**
2. Copy:
   - Project URL
   - `anon` `public` key
   - `service_role` `secret` key (keep secret!)

### 1.4 Configure Row Level Security (RLS)
Schema already includes RLS policies. Verify:
```sql
-- Check RLS is enabled
SELECT schemaname, tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public';
```

---

## 💳 Step 2: Razorpay Setup

### 2.1 Create Account
1. Sign up at [dashboard.razorpay.com](https://dashboard.razorpay.com)
2. Complete KYC (required for production)
3. Enable test mode for initial testing

### 2.2 Get API Keys
1. Go to **Settings** → **API Keys**
2. Generate keys:
   - Test mode: `rzp_test_xxxxx`
   - Live mode: `rzp_live_xxxxx` (after KYC)
3. Copy Key ID and Key Secret

### 2.3 Configure Webhooks
1. Go to **Settings** → **Webhooks**
2. Add webhook URL: `https://api.yourdomain.com/api/webhooks/razorpay`
3. Select events:
   - `payment.captured`
   - `payment.failed`
   - `subscription.activated`
   - `subscription.cancelled`

---

## 🖥️ Step 3: Backend Deployment

### Option A: Railway (Recommended)

#### 3.1 Install Railway CLI
```bash
npm install -g @railway/cli
railway login
```

#### 3.2 Initialize Project
```bash
cd backend
railway init
railway link
```

#### 3.3 Set Environment Variables
```bash
railway variables set SUPABASE_URL=https://xxx.supabase.co
railway variables set SUPABASE_ANON_KEY=eyJxxx
railway variables set SUPABASE_SERVICE_KEY=eyJxxx
railway variables set RAZORPAY_KEY_ID=rzp_live_xxx
railway variables set RAZORPAY_KEY_SECRET=xxx
railway variables set SECRET_KEY=$(openssl rand -hex 32)
railway variables set FRONTEND_URL=https://swingai.vercel.app
railway variables set APP_ENV=production
```

#### 3.4 Add Redis
```bash
railway add
# Select: Redis
```

#### 3.5 Deploy
```bash
railway up
```

#### 3.6 Get Backend URL
```bash
railway open
# Copy the URL (e.g., swingai-backend.up.railway.app)
```

### Option B: Render

#### 3.1 Connect Repository
1. Go to [render.com](https://render.com)
2. Click **New +** → **Web Service**
3. Connect your GitHub repository
4. Select `backend` directory

#### 3.2 Configure Service
- **Name**: swingai-backend
- **Region**: Singapore
- **Branch**: main
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 4`

#### 3.3 Add Environment Variables
Add all variables from `.env.example`

#### 3.4 Add Redis
1. Click **New +** → **Redis**
2. Copy Redis URL
3. Add to backend: `REDIS_URL=<redis-url>`

#### 3.5 Deploy
Click **Create Web Service**

---

## 🌐 Step 4: Frontend Deployment (Vercel)

### 4.1 Install Vercel CLI
```bash
npm install -g vercel
vercel login
```

### 4.2 Deploy Frontend
```bash
cd frontend
vercel --prod
```

### 4.3 Set Environment Variables
Go to Vercel dashboard → Project → Settings → Environment Variables:

```
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJxxx
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_RAZORPAY_KEY_ID=rzp_live_xxx
NEXT_PUBLIC_WS_URL=wss://api.yourdomain.com/ws
```

### 4.4 Redeploy
```bash
vercel --prod
```

---

## 🤖 Step 5: ML Model Deployment (Modal)

### 5.1 Install Modal
```bash
pip install modal
```

### 5.2 Authenticate
```bash
modal token new
```

### 5.3 Upload Trained Models
```bash
# After training in Google Colab
modal volume create swingai-models

# Upload models
modal volume put swingai-models catboost_model.pkl /catboost_model.pkl
modal volume put swingai-models tft_model.pkl /tft_model.pkl
modal volume put swingai-models stockformer_model.pkl /stockformer_model.pkl
```

### 5.4 Deploy Inference Service
```bash
cd ml-deployment
modal deploy modal_inference.py
```

### 5.5 Get Inference URL
```bash
modal app list
# Copy the URL
```

### 5.6 Update Backend
Add to backend environment variables:
```bash
ML_INFERENCE_URL=https://your-model.modal.run
```

---

## 🌍 Step 6: Domain Configuration (Optional)

### 6.1 Frontend Domain
1. In Vercel: Settings → Domains
2. Add your domain: `swingai.com`
3. Update DNS records as instructed

### 6.2 Backend Domain
1. In Railway/Render: Settings → Custom Domains
2. Add subdomain: `api.swingai.com`
3. Update DNS:
   ```
   Type: CNAME
   Name: api
   Value: <railway-url>
   ```

### 6.3 Update Environment Variables
Update `FRONTEND_URL` and `NEXT_PUBLIC_API_URL` with new domains

---

## 🔒 Step 7: Security Hardening

### 7.1 Enable HTTPS
- ✅ Vercel: Auto-enabled
- ✅ Railway/Render: Auto-enabled with custom domain

### 7.2 Set Strong Secrets
```bash
# Generate new secret key
openssl rand -hex 32

# Update in Railway/Render
railway variables set SECRET_KEY=<new-key>
```

### 7.3 Configure CORS
In backend, update allowed origins:
```python
ALLOWED_ORIGINS = [
    "https://swingai.com",
    "https://www.swingai.com"
]
```

### 7.4 Enable Supabase RLS
Already configured in schema. Verify policies are active.

### 7.5 Set Up Sentry (Error Tracking)
```bash
# Add to backend
pip install sentry-sdk[fastapi]

# Set environment variable
SENTRY_DSN=https://xxx@xxx.ingest.sentry.io/xxx
```

---

## 📊 Step 8: Monitoring & Logging

### 8.1 Supabase Monitoring
- Dashboard → Database → Logs
- Set up alerts for high CPU/memory

### 8.2 Railway/Render Monitoring
- Built-in metrics dashboard
- Set up alerts for downtime

### 8.3 Vercel Analytics
- Enable in Vercel dashboard
- Track user traffic and performance

---

## ✅ Step 9: Testing Production

### 9.1 Functional Tests
- [ ] User signup/login
- [ ] Subscription payment (test mode first)
- [ ] View signals on dashboard
- [ ] Execute trade
- [ ] WebSocket real-time updates
- [ ] Stop loss/target alerts

### 9.2 Load Testing
```bash
# Install Apache Bench
brew install httpd  # macOS

# Test backend
ab -n 1000 -c 10 https://api.swingai.com/health

# Test frontend
ab -n 1000 -c 10 https://swingai.com/
```

### 9.3 Security Testing
```bash
# Run security audit
pip install bandit
bandit -r backend/app

# Check dependencies
pip install safety
safety check --file backend/requirements.txt
```

---

## 🚨 Step 10: Go Live Checklist

- [ ] Switch Razorpay to live mode
- [ ] Update Razorpay keys in environment
- [ ] Test live payment with small amount
- [ ] Enable production logging
- [ ] Set up error alerts (Sentry)
- [ ] Configure database backups (Supabase auto-backups)
- [ ] Add monitoring dashboard
- [ ] Prepare customer support email
- [ ] Create terms of service & privacy policy
- [ ] Test all features end-to-end
- [ ] Announce launch! 🎉

---

## 📈 Scaling Considerations

### When to Scale:

**Users < 100**: Current setup sufficient
- Railway Starter ($5/month)
- Vercel Free tier
- Modal pay-as-you-go

**Users 100-1000**: Upgrade tiers
- Railway Pro ($20/month)
- Add Redis clustering
- Increase Modal concurrency

**Users 1000+**: Consider
- AWS/GCP migration
- Kubernetes for backend
- CDN for frontend
- Dedicated database
- Load balancing

---

## 🆘 Troubleshooting

### Backend won't start
```bash
# Check logs
railway logs

# Common issues:
1. Missing environment variables
2. Supabase keys incorrect
3. Python version mismatch (need 3.11)
```

### Frontend build fails
```bash
# Check build logs in Vercel
# Common issues:
1. Missing environment variables
2. Type errors (run `npm run build` locally first)
3. Missing dependencies
```

### WebSocket not connecting
```bash
# Check:
1. CORS settings in backend
2. WS URL correct (wss:// not ws:// in production)
3. Firewall not blocking WebSocket connections
```

### ML predictions failing
```bash
# Check Modal logs
modal app logs swingai-ml-inference

# Common issues:
1. Models not uploaded to volume
2. GPU allocation timeout
3. Incorrect input format
```

---

## 📞 Support

- **Documentation**: `/docs`
- **Issues**: GitHub Issues
- **Email**: support@swingai.com

---

## 🎉 Congratulations!

Your SwingAI platform is now live in production!

Monitor closely for the first few days and be ready to scale as needed.

---

**Next Steps**:
1. Market your platform
2. Gather user feedback
3. Iterate and improve
4. Add more features
5. Scale! 🚀
