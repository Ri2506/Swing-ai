"""
SwingAI Backend Server Entry Point
This file serves as the entry point for the uvicorn server.
It imports the FastAPI app from the actual source location.
"""
import sys
import os

# Add the src directory to the path
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app')

# Import the app from the actual backend location
from src.backend.api.app import app

# Re-export for uvicorn
__all__ = ['app']
