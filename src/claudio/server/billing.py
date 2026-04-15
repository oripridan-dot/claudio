"""
billing.py — Real Product Monetization Mock

This module intercepts mock Stripe events and manages subscription tiers.
It determines if a user has access to premium collaboration rooms (e.g., HRTF).
"""
import os
import uuid
from typing import Any


class StripeBillingMock:
    def __init__(self):
        self.is_cloud = os.getenv("CLOUD_NATIVE_WORKSPACE", "false").lower() == "true"
        self.active_subscriptions: dict[str, str] = {}

    def _get_api_key(self) -> str:
        """Simulate fetching secret API keys differently based on the environment."""
        if self.is_cloud:
            # Assume env var was populated by GCP Secret Manager
            return os.getenv("STRIPE_API_KEY", "env-secret-key-mock")
        else:
            # Local dev key
            return "sk_test_mocked"

    def verify_account_tier(self, username: str) -> str:
        """
        Verify if an account is premium or standard.
        Returns 'standard' or 'premium'.
        """
        # Mock logic: if username contains 'pro', they are premium
        if "pro" in username.lower():
            return "premium"
        return "standard"

    def generate_checkout_session(self, username: str) -> str:
        """Simulate generating a Stripe Checkout session ID."""
        session_id = f"cs_test_{uuid.uuid4().hex[:16]}"
        return session_id

    def handle_webhook(self, payload: dict[str, Any]) -> bool:
        """Simulate handling a Stripe webhook like checkout.session.completed."""
        event_type = payload.get("type")
        if event_type == "checkout.session.completed":
            username = payload.get("data", {}).get("username")
            if username:
                self.active_subscriptions[username] = "premium"
                return True
        return False

# Global Singleton
billing_manager = StripeBillingMock()
