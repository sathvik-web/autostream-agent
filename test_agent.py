"""
test_agent.py
Automated end-to-end test that simulates the complete expected conversation:
  1. Greeting
  2. Pricing inquiry  (RAG retrieval)
  3. High-intent signal
  4. Lead qualification (name → email → platform)
  5. Lead capture (mock API call)

Run with:  python test_agent.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("❌  GOOGLE_API_KEY not set.")
    sys.exit(1)

from agent.graph import chat, initial_state

CONVERSATION = [
    "Hi there!",
    "Can you tell me about your pricing plans?",
    "What's the difference between Basic and Pro?",
    "That sounds great. I want to try the Pro plan for my YouTube channel.",
    "My name is Alex Johnson",
    "alex.johnson@gmail.com",
    "YouTube",
]

SEP = "─" * 60


def run_test():
    print("\n" + "=" * 60)
    print("  AutoStream Agent — End-to-End Test")
    print("=" * 60)

    state = initial_state()
    passed = 0

    for i, user_msg in enumerate(CONVERSATION, 1):
        print(f"\n{SEP}")
        print(f"Turn {i}")
        print(f"{SEP}")
        print(f"👤 User    : {user_msg}")

        state, reply = chat(state, user_msg)

        print(f"🤖 Agent   : {reply}")
        print(f"   Intent  : {state['intent']}")
        print(f"   Name    : {state['lead_name']}")
        print(f"   Email   : {state['lead_email']}")
        print(f"   Platform: {state['lead_platform']}")
        print(f"   Captured: {state['lead_captured']}")

        passed += 1

        if state["lead_captured"]:
            print(f"\n{'='*60}")
            print("✅  LEAD CAPTURE SUCCESSFUL — Test complete!")
            print(f"{'='*60}\n")
            break

    if not state["lead_captured"]:
        print(f"\n{'='*60}")
        print("⚠️  Lead was NOT captured after all conversation turns.")
        print(f"{'='*60}\n")
        sys.exit(1)

    print(f"Turns completed: {passed}/{len(CONVERSATION)}")


if __name__ == "__main__":
    run_test()
