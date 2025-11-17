"""Vercel serverless entrypoint for the Flask application.

Vercel's Python runtime looks for a module-level `app` variable that exposes a
WSGI application. We simply import the existing Flask instance from `main`.
"""
from main import app as flask_app

# Vercel expects this name; we re-export the Flask instance created in main.py.
app = flask_app
