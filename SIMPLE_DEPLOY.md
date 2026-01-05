# 🚀 SwingAI - Super Simple Deployment Guide

**No Docker. No Kubernetes. Just Vercel + Railway + Modal + Supabase.**

---

## 🎯 **Your Stack**

| Service | What For | Deploy Time |
|---------|----------|-------------|
| **Supabase** | Database + Auth | 5 min |
| **Railway** | Backend API | 5 min |
| **Vercel** | Frontend | 3 min |
| **Modal** | AI Models | 5 min |

**Total Time**: ~20 minutes! ⚡

---

## 1️⃣ **Supabase Setup** (5 min)

### Step 1: Create Project
1. Go to [supabase.com](https://supabase.com)
2. Click "New Project"
3. Name: `swingai`
4. Region: **Singapore** (closest to India)
5. Database password: (save this!)

### Step 2: Run Database Schema
1. Go to **SQL Editor**
2. Copy everything from `infrastructure/database/complete_schema.sql`
3. Paste and click **RUN**
4. Wait for "Success" ✅

### Step 3: Get API Keys
1. Go to **Settings** → **API**
2. Copy these 3 things:
   - Project URL: `https://xxx.supabase.co`
   - `anon` `public` key: `eyJxxx...`
   - `service_role` key: `eyJxxx...` (keep secret!)

✅ **Done!** Supabase is ready.

---

## 2️⃣ **Railway Backend** (5 min)

### Step 1: Install Railway CLI
```bash
npm install -g @railway/cli
railway login
```

### Step 2: Deploy Backend
```bash
cd /Users/rishi/Downloads/SwingAI

# Initialize Railway project
railway init

# Add variables
railway variables set SUPABASE_URL="https://xxx.supabase.co"
railway variables set SUPABASE_ANON_KEY="eyJxxx"
railway variables set SUPABASE_SERVICE_KEY="eyJxxx"
railway variables set RAZORPAY_KEY_ID="rzp_test_xxx"
railway variables set RAZORPAY_KEY_SECRET="xxx"
railway variables set SECRET_KEY="$(openssl rand -hex 32)"
railway variables set APP_ENV="production"
railway variables set FRONTEND_URL="https://swingai.vercel.app"

# Deploy!
railway up
```

### Step 3: Get Backend URL
```bash
railway open
# Copy the URL (e.g., swingai-production.up.railway.app)
```

✅ **Done!** Backend is live at Railway.

---

## 3️⃣ **Vercel Frontend** (3 min)

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
vercel login
```

### Step 2: Deploy Frontend
```bash
cd src/frontend

# Deploy
vercel --prod
```

### Step 3: Add Environment Variables
In Vercel dashboard → Settings → Environment Variables:
```
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJxxx
NEXT_PUBLIC_API_URL=https://swingai-production.up.railway.app
NEXT_PUBLIC_RAZORPAY_KEY_ID=rzp_test_xxx
```

### Step 4: Redeploy
```bash
vercel --prod
```

✅ **Done!** Frontend is live at Vercel.

---

## 4️⃣ **Modal AI Models** (5 min)

### Step 1: Install Modal
```bash
pip install modal
modal token new
```

### Step 2: Upload Models
```bash
# After training in Google Colab, download models
# Then upload to Modal

modal volume create swingai-models

# Upload your trained models
modal volume put swingai-models catboost_model.pkl /catboost_model.pkl
modal volume put swingai-models tft_model.pth /tft_model.pth
modal volume put swingai-models stockformer_model.pth /stockformer_model.pth
```

### Step 3: Deploy Inference
```bash
cd ml/inference
modal deploy modal_inference.py
```

### Step 4: Get Inference URL
```bash
modal app list
# Copy the URL
```

### Step 5: Add to Railway
```bash
railway variables set ML_INFERENCE_URL="https://your-app.modal.run"
```

✅ **Done!** AI models are live on Modal.

---

## 🎉 **YOU'RE LIVE!**

Your app is now deployed:
- **Frontend**: https://swingai.vercel.app
- **Backend**: https://swingai-production.up.railway.app
- **Database**: Supabase
- **AI Models**: Modal

---

## 💰 **Monthly Cost**

- **Supabase**: $0 (Free tier - 500MB database)
- **Railway**: $5 (Starter plan - 500 hours)
- **Vercel**: $0 (Hobby tier - unlimited bandwidth)
- **Modal**: ~$20 (Pay per use - GPU)

**Total**: **~$25/month** to start! 🎯

---

## 🔄 **Making Updates**

### Update Backend
```bash
# Just push changes
railway up
```

### Update Frontend
```bash
cd src/frontend
vercel --prod
```

### Update AI Models
```bash
cd ml/inference
modal deploy modal_inference.py
```

**That's it!** No Docker, no K8s, just simple deployments.

---

## 🆘 **Troubleshooting**

### Backend won't start on Railway
```bash
# Check logs
railway logs

# Common fix: Make sure requirements.txt is in root
```

### Frontend build fails on Vercel
```bash
# Check build logs in Vercel dashboard
# Common fix: Make sure package.json is in src/frontend/
```

### Modal deployment fails
```bash
# Check you're authenticated
modal token new

# Make sure models are uploaded
modal volume list
```

---

## ✅ **Verification Checklist**

After deployment:
- [ ] Visit your Vercel URL - should see landing page
- [ ] Go to /api/health on Railway - should return {"status": "healthy"}
- [ ] Test signup on frontend
- [ ] Check Supabase dashboard - should see new user
- [ ] Create a test payment (Razorpay test mode)
- [ ] Check Railway logs - no errors

---

**That's it! Simple, fast, no Docker needed!** 🚀

For detailed Railway/Vercel configs, they auto-detect Python/Next.js projects!
