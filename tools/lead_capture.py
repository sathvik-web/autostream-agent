"""
tools/lead_capture.py
Mock lead capture tool for AutoStream.
In production, this would call a CRM API (HubSpot, Salesforce, etc.)
"""

import json
from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Captures a qualified lead after collecting all required information.

    Args:
        name:     Full name of the prospect
        email:    Email address of the prospect
        platform: Creator platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        dict with status and lead_id
    """
    # Simulate CRM record creation
    lead_id = f"LEAD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    lead_record = {
        "lead_id": lead_id,
        "name": name,
        "email": email,
        "platform": platform,
        "source": "AutoStream Conversational Agent",
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "status": "new",
    }

    # ── Console output (visible in terminal / logs) ───────────────────────
    print("\n" + "=" * 60)
    print("✅  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Lead ID  : {lead_id}")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"  Time     : {lead_record['captured_at']}")
    print("=" * 60 + "\n")

    return {"success": True, "lead_id": lead_id, "record": lead_record}
