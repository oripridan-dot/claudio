"""
billing.py — Real Product Monetization

This module manages subscription tiers and interacts with Stripe.
It determines if a user has access to premium collaboration rooms (e.g., HRTF).
"""

import os
import uuid
from typing import Any

class StripeBillingManager:
    def __init__(self):
        # Enforce Validity-First: Read environment directly without hallucinating success.
        self.is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
        self.active_subscriptions: dict[str, str] = {}
        self.stripe = None
        
        if self.is_production:
            import stripe
            self.stripe = stripe
            api_key = os.getenv("STRIPE_API_KEY")
            if not api_key:
                raise ValueError("ValidityError: STRIPE_API_KEY is missing. Cannot start billing in production.")
            self.stripe.api_key = api_key

    def verify_account_tier(self, username: str) -> str:
        """
        Verify if an account is premium or standard.
        """
        if not self.is_production:
            return "premium" if "pro" in username.lower() else "standard"
            
        # Real verification would happen via active_subscriptions cache updated by webhooks
        return self.active_subscriptions.get(username, "standard")

    def generate_checkout_session(self, username: str) -> str:
        """Generate a Stripe Checkout session ID."""
        if not self.is_production:
            return f"cs_test_{uuid.uuid4().hex[:16]}"
            
        if not self.stripe:
            raise RuntimeError("ValidityError: Stripe module not loaded.")

        session = self.stripe.checkout.Session.create(
            payment_method_types=['card'],
            mode='subscription',
            success_url='https://example.com/success',
            cancel_url='https://example.com/cancel',
            metadata={'username': username}
        )
        return session.id

    def handle_webhook(self, payload: dict[str, Any]) -> bool:
        """Handle checkout.session.completed webhook."""
        if payload.get("type") == "checkout.session.completed":
            data_obj = payload.get("data", {}).get("object", {})
            # Look in standard stripe nested payload or fallback to dev payload format
            username = data_obj.get("metadata", {}).get("username")
            if not username:
                username = payload.get("data", {}).get("username")
            if username:
                self.active_subscriptions[username] = "premium"
                return True
        return False


# Global Singleton
billing_manager = StripeBillingManager()
