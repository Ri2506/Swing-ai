"""
Core configuration for SwingAI backend
Centralized settings with environment variable management
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation and environment variable loading"""

    # ============================================================================
    # APPLICATION
    # ============================================================================
    APP_NAME: str = "SwingAI"
    APP_VERSION: str = "2.0.0"
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")

    # ============================================================================
    # API
    # ============================================================================
    API_PREFIX: str = "/api"
    API_VERSION: str = "v1"

    # ============================================================================
    # FRONTEND
    # ============================================================================
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # ============================================================================
    # SUPABASE
    # ============================================================================
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    # ============================================================================
    # RAZORPAY
    # ============================================================================
    RAZORPAY_KEY_ID: str = os.getenv("RAZORPAY_KEY_ID", "")
    RAZORPAY_KEY_SECRET: str = os.getenv("RAZORPAY_KEY_SECRET", "")

    # ============================================================================
    # REDIS
    # ============================================================================
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")

    # ============================================================================
    # CORS
    # ============================================================================
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://swingai.vercel.app",
        "https://*.vercel.app",
    ]

    # ============================================================================
    # RATE LIMITING
    # ============================================================================
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))

    # ============================================================================
    # ML MODEL
    # ============================================================================
    ML_INFERENCE_URL: str = os.getenv("ML_INFERENCE_URL", "")
    ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", "ml/models")

    # ============================================================================
    # MONITORING
    # ============================================================================
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ============================================================================
    # TELEGRAM (Optional)
    # ============================================================================
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")

    # ============================================================================
    # WEBSOCKET
    # ============================================================================
    WS_MESSAGE_QUEUE_SIZE: int = 100
    WS_HEARTBEAT_INTERVAL: int = 30

    # ============================================================================
    # TRADING
    # ============================================================================
    MARKET_OPEN_TIME: str = "09:15"
    MARKET_CLOSE_TIME: str = "15:30"
    PRE_MARKET_SCAN_TIME: str = "08:30"
    POST_MARKET_PROCESS_TIME: str = "16:00"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
