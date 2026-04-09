"""
auth.py — JWT Authentication Module

Provides stateless token verification used to restrict WebSocket gateways to
authenticated musicians, preventing abuse of open ports.
"""

from __future__ import annotations

import os
import secrets
import time

import jwt

# Get secret from environment, or generate an ephemeral one for local dev instances
JWT_SECRET = os.environ.get("CLAUDIO_JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_SECONDS = 24 * 3600  # 24 hours


def create_token(username: str, role: str = "musician") -> str:
    """Generate a signed JWT for the given username."""
    now = int(time.time())
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now + JWT_EXPIRATION_SECONDS,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict | None:
    """Verify and decode a JWT. Returns payload dict if valid, None if invalid/expired."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.InvalidTokenError:
        return None
