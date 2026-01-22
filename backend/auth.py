"""
🔐 GOOGLE AUTH API
==================
Emergent-managed Google OAuth authentication.

Flow:
1. User clicks "Sign in with Google" 
2. Redirected to auth.emergentagent.com
3. After Google auth, redirected back with session_id
4. Frontend sends session_id to backend
5. Backend exchanges for user data & creates session
"""

from fastapi import APIRouter, HTTPException, Response, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone, timedelta
import httpx
import uuid

from database import get_supabase

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Emergent Auth API
EMERGENT_AUTH_URL = "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data"

# Session duration
SESSION_DURATION_DAYS = 7


# ============================================================
# 📝 DATA MODELS
# ============================================================

class SessionRequest(BaseModel):
    """Request to exchange session_id for user data"""
    session_id: str


class UserResponse(BaseModel):
    """User data response"""
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    paper_balance: float = 1000000.0
    is_paper_mode: bool = True


# ============================================================
# 🔧 HELPER FUNCTIONS
# ============================================================

def generate_user_id() -> str:
    """Generate a custom user ID"""
    return f"user_{uuid.uuid4().hex[:12]}"


async def get_or_create_user(email: str, name: str, picture: str = None) -> dict:
    """Get existing user or create new one"""
    db = get_supabase()
    if not db:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    # Check if user exists
    result = db.table("users").select("*").eq("email", email).execute()
    
    if result.data and len(result.data) > 0:
        # User exists - update name/picture if changed
        user = result.data[0]
        if user.get("name") != name or user.get("picture") != picture:
            db.table("users").update({
                "name": name,
                "picture": picture
            }).eq("email", email).execute()
            user["name"] = name
            user["picture"] = picture
        return user
    
    # Create new user
    new_user = {
        "email": email,
        "name": name,
        "picture": picture,
        "paper_balance": 1000000.0,  # ₹10 Lakh starting balance
        "is_paper_mode": True,
    }
    
    result = db.table("users").insert(new_user).execute()
    
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create user")
    
    return result.data[0]


async def create_session(user_id: str, session_token: str) -> dict:
    """Create a new session in database"""
    db = get_supabase()
    if not db:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_DURATION_DAYS)
    
    session = {
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": expires_at.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    
    result = db.table("user_sessions").insert(session).execute()
    return result.data[0] if result.data else None


async def get_session(session_token: str) -> dict:
    """Get session by token"""
    db = get_supabase()
    if not db:
        return None
    
    result = db.table("user_sessions").select("*").eq("session_token", session_token).execute()
    
    if not result.data:
        return None
    
    session = result.data[0]
    
    # Check expiry
    expires_at = session.get("expires_at")
    if expires_at:
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at < datetime.now(timezone.utc):
            # Session expired - delete it
            db.table("user_sessions").delete().eq("session_token", session_token).execute()
            return None
    
    return session


async def delete_session(session_token: str):
    """Delete a session"""
    db = get_supabase()
    if db:
        db.table("user_sessions").delete().eq("session_token", session_token).execute()


def get_session_token_from_request(request: Request) -> Optional[str]:
    """Extract session token from cookie or Authorization header"""
    # First try cookie
    session_token = request.cookies.get("session_token")
    if session_token:
        return session_token
    
    # Then try Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]
    
    return None


# ============================================================
# 🎯 API ENDPOINTS
# ============================================================

@router.post("/session")
async def exchange_session(request: SessionRequest, response: Response):
    """
    Exchange session_id from Emergent Auth for user data.
    
    This is called after Google OAuth redirect with session_id in URL.
    """
    try:
        # Call Emergent Auth API to get user data
        async with httpx.AsyncClient() as client:
            auth_response = await client.get(
                EMERGENT_AUTH_URL,
                headers={"X-Session-ID": request.session_id},
                timeout=10.0
            )
        
        if auth_response.status_code != 200:
            raise HTTPException(
                status_code=401, 
                detail="Invalid session_id or authentication failed"
            )
        
        auth_data = auth_response.json()
        
        # Extract user data
        email = auth_data.get("email")
        name = auth_data.get("name", email.split("@")[0] if email else "User")
        picture = auth_data.get("picture")
        emergent_session_token = auth_data.get("session_token")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not provided by auth")
        
        # Get or create user in our database
        user = await get_or_create_user(email, name, picture)
        user_id = user["id"]
        
        # Create session in our database
        await create_session(user_id, emergent_session_token)
        
        # Set httpOnly cookie
        response.set_cookie(
            key="session_token",
            value=emergent_session_token,
            httponly=True,
            secure=True,
            samesite="none",
            path="/",
            max_age=SESSION_DURATION_DAYS * 24 * 60 * 60  # 7 days in seconds
        )
        
        return {
            "success": True,
            "user": {
                "user_id": user_id,
                "email": user["email"],
                "name": user["name"],
                "picture": user.get("picture"),
                "paper_balance": float(user.get("paper_balance", 1000000)),
                "is_paper_mode": user.get("is_paper_mode", True),
            }
        }
        
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Auth service unavailable: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Auth error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me")
async def get_current_user(request: Request):
    """
    Get current authenticated user.
    
    Checks session_token from cookie or Authorization header.
    """
    session_token = get_session_token_from_request(request)
    
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Get session
    session = await get_session(session_token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    # Get user
    db = get_supabase()
    if not db:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    result = db.table("users").select("*").eq("id", session["user_id"]).execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = result.data[0]
    
    return {
        "user_id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "picture": user.get("picture"),
        "paper_balance": float(user.get("paper_balance", 1000000)),
        "is_paper_mode": user.get("is_paper_mode", True),
    }


@router.post("/logout")
async def logout(request: Request, response: Response):
    """
    Logout - delete session and clear cookie.
    """
    session_token = get_session_token_from_request(request)
    
    if session_token:
        await delete_session(session_token)
    
    # Clear cookie
    response.delete_cookie(
        key="session_token",
        path="/",
        secure=True,
        samesite="none"
    )
    
    return {"success": True, "message": "Logged out successfully"}


@router.get("/check")
async def check_auth(request: Request):
    """
    Quick auth check - returns whether user is authenticated.
    
    Used by frontend to check auth status without full user data.
    """
    session_token = get_session_token_from_request(request)
    
    if not session_token:
        return {"authenticated": False}
    
    session = await get_session(session_token)
    
    return {"authenticated": session is not None}
