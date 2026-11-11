"""
test_stepped_care_verification.py - End-to-end verification of strict stepped-care thresholds.
Tests all 4 tiers and proves:
1. "feeling some demotivation" (PHQ=1) stays strictly in TIER_1_SELF_CARE (CONVERSATIONAL_INTAKE).
2. Moderate anxiety/depression (PHQ/GAD in [5..14]) routes to TIER_2_MILD_MOD (INITIATE_TELEHEALTH_HANDOFF).
3. Somatic cardiac red flag or PHQ/GAD >= 15 routes to TIER_3_SEVERE (TRIGGER_JIT_IDENTITY_GATE).
4. Suicidal intent routes to TIER_EMERGENCY (HARD_EMERGENCY_OVERRIDE).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import uuid
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db

init_db()
client = TestClient(app)


def test_demotivation_phq1_remains_level_1():
    """Confirms single-item low score (demotivation, PHQ=1) does NOT escalate to In-Clinic JIT."""
    session_id = f"test_demot_{uuid.uuid4().hex[:8]}"
    with client.websocket_connect(f"/ws/triage/{session_id}") as ws:
        ws.send_json({
            "transcript": "feeling some demotivation",
            "arousal": 0.50,
            "valence": 0.50,
            "language": "en-US"
        })
        resp = ws.receive_json()
        assert resp["type"] == "AGENT_TURN"
        t = resp["dashboard_telemetry"]
        action = resp["dispatched_action"]

        print("\n--- Demotivation (PHQ=1) Response ---")
        print("Triage Tier:", t["triage_tier"])
        print("Active Mode:", t["active_mode"])
        print("PHQ-9 Total:", t["phq9_total"])
        print("GAD-7 Total:", t["gad7_total"])
        print("Somatic Count:", t["somatic_count"])
        print("Dispatched Action:", action["action_type"])
        print("Agent Reply:", resp["reply_text"])

        # Strict Assertions:
        assert t["triage_tier"] == "TIER_1_SELF_CARE", f"Expected TIER_1_SELF_CARE but got {t['triage_tier']}"
        assert t["somatic_count"] == 0, f"Expected 0 somatic symptoms but got {t['somatic_count']}"
        assert action["action_type"] in ("CONVERSATIONAL_INTAKE", "SWITCH_UI_MODE")
        assert action["action_type"] != "TRIGGER_JIT_IDENTITY_GATE", "ERR: Premature in-clinic JIT gate triggered!"
        assert action["action_type"] != "INITIATE_TELEHEALTH_HANDOFF", "ERR: Premature telehealth gate triggered!"


def test_level_2_moderate_telehealth():
    """Confirms PHQ or GAD in [5..14] without somatic red flags routes to Telehealth handoff upon screening/booking request."""
    session_id = f"test_lvl2_{uuid.uuid4().hex[:8]}"
    with client.websocket_connect(f"/ws/triage/{session_id}") as ws:
        # Provide explicit moderate anxiety symptoms across multiple GAD-7 questions and request consultation:
        ws.send_json({
            "transcript": "Nearly every day for the past month I feel intensely nervous, I cannot stop or control my constant worrying, and I am so restless I cannot sit still or relax at all. Can I book a consultation with a doctor?",
            "arousal": 0.65,
            "valence": 0.35,
            "language": "en-US"
        })
        resp = ws.receive_json()
        assert resp["type"] == "AGENT_TURN"
        t = resp["dashboard_telemetry"]
        action = resp["dispatched_action"]

        print("\n--- Level 2 Moderate Response ---")
        print("Triage Tier:", t["triage_tier"])
        print("PHQ-9 Total:", t["phq9_total"])
        print("GAD-7 Total:", t["gad7_total"])
        print("Somatic Count:", t["somatic_count"])
        print("Dispatched Action:", action["action_type"])

        # Strict Assertions for Level 2:
        assert t["triage_tier"] == "TIER_2_MILD_MOD"
        assert action["action_type"] == "INITIATE_TELEHEALTH_HANDOFF"
        assert "room_link" in action["payload"]
        assert action["payload"]["room_link"] == "https://telehealth.seha.ae/room/virtual-care"


def test_level_3_cardiac_somatic_in_clinic():
    """Confirms explicit chest tightness triggers TIER_3_SEVERE and Al Bateen in-clinic booking."""
    session_id = f"test_cardiac_{uuid.uuid4().hex[:8]}"
    with client.websocket_connect(f"/ws/triage/{session_id}") as ws:
        ws.send_json({
            "transcript": "I have constant chest tightness, racing heart, and feel completely exhausted.",
            "arousal": 0.85,
            "valence": 0.20,
            "language": "en-US"
        })
        resp = ws.receive_json()
        assert resp["type"] == "AGENT_TURN"
        t = resp["dashboard_telemetry"]
        action = resp["dispatched_action"]

        print("\n--- Level 3 Cardiac Somatic Response ---")
        print("Triage Tier:", t["triage_tier"])
        print("Somatic Count:", t["somatic_count"])
        print("Dispatched Action:", action["action_type"])
        print("Facility:", action["payload"].get("facility"))

        assert t["triage_tier"] == "TIER_3_SEVERE"
        assert t["somatic_count"] >= 1
        assert action["action_type"] == "TRIGGER_JIT_IDENTITY_GATE"
        assert "Al Bateen" in action["payload"].get("facility", "")


def test_emergency_override():
    """Confirms suicidal intent triggers TIER_EMERGENCY and locks session."""
    session_id = f"test_emerg_{uuid.uuid4().hex[:8]}"
    with client.websocket_connect(f"/ws/triage/{session_id}") as ws:
        ws.send_json({
            "transcript": "I swallowed all my pills to end it and I want to die.",
            "arousal": 0.90,
            "valence": 0.10,
            "language": "en-US"
        })
        resp = ws.receive_json()
        assert resp["type"] == "AGENT_TURN"
        t = resp["dashboard_telemetry"]
        action = resp["dispatched_action"]

        print("\n--- Emergency Response ---")
        print("Triage Tier:", t["triage_tier"])
        print("Active Mode:", t["active_mode"])
        print("Dispatched Action:", action["action_type"])

        assert t["triage_tier"] == "TIER_EMERGENCY"
        assert t["active_mode"] == "EMERGENCY_LOCK"
        assert action["action_type"] == "HARD_EMERGENCY_OVERRIDE"


if __name__ == "__main__":
    test_demotivation_phq1_remains_level_1()
    test_level_2_moderate_telehealth()
    test_level_3_cardiac_somatic_in_clinic()
    test_emergency_override()
    print("\nALL STEPPED-CARE VERIFICATION TESTS PASSED SUCCESSFULLY!")
