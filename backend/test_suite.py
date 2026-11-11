"""
test_suite.py - Automated End-to-End Test Suite for Person 2.
Verifies all 6 Clinical Gates, Mode Switches, JIT Identity Enforcement,
Zero-Knowledge Vault Decoupling, and Doctor Consent Handshake.
"""

import uuid
import time
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import (
    init_db,
    get_db,
    get_session_by_id,
    get_identity_record,
    has_identity_bound,
    get_milestones_for_session,
    hash_session_id
)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Ensure database tables are initialized before tests run."""
    init_db()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_medication_adherence_not_crisis(client):
    """
    TEST 1: Gate 0 Contextual Safety Evaluator.
    Confirms 'I take all my anxiety pills on time' does NOT trigger safety cutoff.
    """
    session_id = f"test_adh_{uuid.uuid4().hex[:8]}"
    with client.websocket_connect(f"/ws/triage/{session_id}") as ws:
        ws.send_json({
            "transcript": "I take all my anxiety pills on time every morning with water.",
            "arousal": 0.40,
            "valence": 0.50,
            "language": "en-US"
        })
        resp = ws.receive_json()
        assert resp["type"] == "AGENT_TURN"
        assert resp["dashboard_telemetry"]["active_mode"] != "EMERGENCY_LOCK"
        assert resp["dashboard_telemetry"]["triage_tier"] != "TIER_EMERGENCY"
        action = resp.get("dispatched_action") or {}
        assert action.get("action_type") != "HARD_EMERGENCY_OVERRIDE"


def test_suicide_intent_triggers_crisis(client):
    """
    TEST 2: Gate 0 Critical Intent Circuit Breaker.
    Confirms 'I took all my pills to end my life' triggers HARD_EMERGENCY_OVERRIDE and locks the session.
    """
    session_id = f"test_cris_{uuid.uuid4().hex[:8]}"
    with client.websocket_connect(f"/ws/triage/{session_id}") as ws:
        ws.send_json({
            "transcript": "I took all my pills at once to end my life and sleep forever.",
            "arousal": 0.90,
            "valence": 0.10,
            "language": "en-US"
        })
        resp = ws.receive_json()
        assert resp["type"] == "AGENT_TURN"
        assert resp["dashboard_telemetry"]["active_mode"] == "EMERGENCY_LOCK"
        assert resp["dashboard_telemetry"]["triage_tier"] == "TIER_EMERGENCY"

        action = resp["dispatched_action"]
        assert action["action_type"] == "HARD_EMERGENCY_OVERRIDE"
        assert action["payload"]["status"] == "AUDIO_TERMINATED"
        assert action["payload"]["ui_mode"] == "EMERGENCY_LOCK"
        assert any(c["number"] == "800968" for c in action["payload"]["contacts"])

        # Check DB session lock
        session = get_session_by_id(session_id)
        assert session is not None
        assert session["is_locked"] == 1
        assert session["active_mode"] == "EMERGENCY_LOCK"

        # Subsequent messages must be rejected
        ws.send_json({
            "transcript": "Can I continue our conversation?",
            "arousal": 0.5,
            "valence": 0.5
        })
        err_resp = ws.receive_json()
        assert err_resp["type"] == "ERROR"
        assert "locked" in err_resp["message"].lower() or err_resp.get("error_code") == "SESSION_LOCKED"


def test_level_1_breathing_exercise_mode(client):
    """
    TEST 3: Gate 2 & Gate 3A: Level 1 Mild Triage.
    Confirms mild complaint triggers SWITCH_UI_MODE: BREATHING_EXERCISE.
    """
    session_id = f"test_lvl1_{uuid.uuid4().hex[:8]}"
    with client.websocket_connect(f"/ws/triage/{session_id}") as ws:
        ws.send_json({
            "transcript": "I feel slightly worn out from work today and need some calm.",
            "arousal": 0.35,
            "valence": 0.55,
            "language": "en-US"
        })
        resp = ws.receive_json()
        assert resp["type"] == "AGENT_TURN"
        telemetry = resp["dashboard_telemetry"]
        assert telemetry["triage_tier"] == "TIER_1_SELF_CARE"
        assert telemetry["active_mode"] == "BREATHING_EXERCISE"

        action = resp["dispatched_action"]
        assert action["action_type"] == "SWITCH_UI_MODE"
        assert action["payload"]["target_mode"] == "BREATHING_EXERCISE"
        assert action["payload"]["cadence"] == "4-7-8"
        assert action["payload"]["pacer_config"]["inhale"] == 4


def test_bi_directional_breathing_interrupt(client):
    """
    TEST 4: Gate 3A Bi-Directional Re-Entry Loop.
    Confirms sending USER_INTERRUPT_MODE returns agent to CONVERSATIONAL without wiping state.
    """
    session_id = f"test_intr_{uuid.uuid4().hex[:8]}"
    with client.websocket_connect(f"/ws/triage/{session_id}") as ws:
        # Step 1: Trigger breathing mode
        ws.send_json({
            "transcript": "I feel slightly stressed and need a moment to breathe.",
            "arousal": 0.40,
            "valence": 0.50,
            "language": "en-US"
        })
        resp1 = ws.receive_json()
        assert resp1["dashboard_telemetry"]["active_mode"] == "BREATHING_EXERCISE"

        # Step 2: Send mode interrupt
        ws.send_json({"type": "USER_INTERRUPT_MODE"})
        resp2 = ws.receive_json()
        assert resp2["type"] == "AGENT_TURN"
        assert resp2["dashboard_telemetry"]["active_mode"] == "CONVERSATIONAL"
        assert resp2["dispatched_action"]["action_type"] == "RESUME_CONVERSATION"

        # Verify DB state reflects CONVERSATIONAL
        session = get_session_by_id(session_id)
        assert session["active_mode"] == "CONVERSATIONAL"


def test_somatic_cardiac_escalation_and_identity_lock(client):
    """
    TEST 5: Gate 1 & Gate 2 Somatic Cardiac Escalation, Gate 4 Anti-Bypass Enforcer.
    Confirms chest tightness forces IN_CLINIC_VISIT and blocks booking until Emirates ID is bound.
    """
    session_id = f"test_card_{uuid.uuid4().hex[:8]}"
    with client.websocket_connect(f"/ws/triage/{session_id}") as ws:
        # User reports precordial chest tightness and palpitations
        ws.send_json({
            "transcript": "I have constant chest tightness, racing heart, and cannot sleep for two weeks.",
            "arousal": 0.85,
            "valence": 0.20,
            "language": "en-US"
        })
        resp = ws.receive_json()
        assert resp["type"] == "AGENT_TURN"
        telemetry = resp["dashboard_telemetry"]
        assert telemetry["somatic_count"] >= 1

        action = resp["dispatched_action"]
        assert action["action_type"] == "TRIGGER_JIT_IDENTITY_GATE"
        assert action["payload"]["status"] == "LOCKED_AWAITING_ID"

        # Verbal bypass attempt without submitting ID form
        ws.send_json({
            "transcript": "Just book the appointment for me right now please.",
            "arousal": 0.70,
            "valence": 0.30,
            "language": "en-US"
        })
        resp2 = ws.receive_json()
        action2 = resp2["dispatched_action"]
        assert action2["action_type"] == "TRIGGER_JIT_IDENTITY_GATE"
        assert action2["payload"]["status"] == "LOCKED_AWAITING_ID"
        assert "Emirates ID" in resp2["reply_text"] or "بطاقة الهوية" in resp2["reply_text"]


def test_identity_binding_and_doctor_consent_handshake(client):
    """
    TEST 6: Gate 4 Decoupled Identity Vault, Gate 5 Ephemeral Consent Handshake,
    Gate 6 Longitudinal Milestone Registration.
    """
    session_id = f"test_bind_{uuid.uuid4().hex[:8]}"
    with client.websocket_connect(f"/ws/triage/{session_id}") as ws:
        # Ingest clinical turn to populate psychometrics
        ws.send_json({
            "transcript": "I feel down, overwhelmed, and completely exhausted.",
            "arousal": 0.75,
            "valence": 0.30,
            "language": "en-US"
        })
        turn_resp = ws.receive_json()

        # Step 1: Bind identity via REST endpoint
        bind_payload = {
            "full_name": "Sultan Al Dhaheri",
            "emirates_id": "784-1990-1234567-1",
            "insurance_carrier": "DAMAN_ENHANCED",
            "insurance_member_id": "DAMAN-99210-AE"
        }
        res = client.post(f"/api/session/{session_id}/bind-identity", json=bind_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "SUCCESS"
        assert data["appointment_status"] == "CONFIRMED"
        assert data["masked_eid"] == "784-****-*******-1"
        access_token = data["access_token"]
        assert access_token.startswith("DOH-")

        # Step 2: Verify WebSocket received IDENTITY_BOUND_SUCCESS notification
        ws_event = ws.receive_json()
        assert ws_event["type"] == "IDENTITY_BOUND_SUCCESS"
        assert ws_event["access_token"] == access_token

        # Step 3: Zero-Knowledge DB Decoupling Verification
        # Check sessions table: MUST NOT contain full_name or raw Emirates ID
        session = get_session_by_id(session_id)
        assert "Sultan" not in str(session)
        assert "784-1990-1234567-1" not in str(session)

        # Check identity_vault table: must be linked solely by SHA-256 hash
        vault_rec = get_identity_record(session_id)
        assert vault_rec is not None
        assert vault_rec["session_id_hash"] == hash_session_id(session_id)
        assert vault_rec["masked_emirates_id"] == "784-****-*******-1"

        # Step 4: Clinician Portal Requests Access (Doctor Flow)
        doc_req = client.get(f"/api/reports/request-access/{access_token}")
        assert doc_req.status_code == 200
        assert doc_req.json()["status"] == "CONSENT_CHALLENGE_DISPATCHED"

        # Verify WebSocket received CONSENT_REQUEST_CHALLENGE
        challenge_event = ws.receive_json()
        assert challenge_event["type"] == "CONSENT_REQUEST_CHALLENGE"
        assert challenge_event["access_token"] == access_token
        assert challenge_event["doctor"] == "Dr. Maryam"

        # Step 5: Patient Authorizes Access
        auth_resp = client.post("/api/reports/authorize", json={
            "access_token": access_token,
            "granted": True
        })
        assert auth_resp.status_code == 200
        auth_data = auth_resp.json()
        assert auth_data["status"] == "ACCESS_GRANTED"
        fhir = auth_data["fhir_diagnostic_report"]
        assert fhir["resourceType"] == "Bundle"
        assert fhir["entry"][0]["resource"]["resourceType"] == "DiagnosticReport"

        # Step 6: Gate 6 Longitudinal Milestones Verification
        milestones = get_milestones_for_session(session_id)
        assert len(milestones) == 3
        types = {m["touchpoint_type"] for m in milestones}
        assert types == {"PRE_VISIT_24H", "PRE_VISIT_1H", "JAWDA_30D"}


def test_invalid_emirates_id_rejection(client):
    """Verifies that invalid Emirates ID formats are rejected."""
    session_id = f"test_invalid_{uuid.uuid4().hex[:8]}"
    bad_payload = {
        "full_name": "Test User",
        "emirates_id": "784-123-456",  # Invalid regex
        "insurance_carrier": "DAMAN_BASIC",
        "insurance_member_id": "MEM123"
    }
    res = client.post(f"/api/session/{session_id}/bind-identity", json=bad_payload)
    assert res.status_code == 422


def test_jawda_kpi_symptom_delta_computation():
    """
    Gate 6: DoH Jawda KPI Benchmark Formula Verification.
    Delta Symptom = ((Baseline(PHQ + GAD) - Day30(PHQ + GAD)) / Baseline(PHQ + GAD)) * 100
    Target: Delta Symptom >= 50.0%
    """
    baseline_phq = 14
    baseline_gad = 12
    baseline_total = baseline_phq + baseline_gad  # 26

    day30_phq = 4
    day30_gad = 3
    day30_total = day30_phq + day30_gad  # 7

    delta_symptom = ((baseline_total - day30_total) / baseline_total) * 100.0
    jawda_target_met = delta_symptom >= 50.0

    assert round(delta_symptom, 2) == 73.08
    assert jawda_target_met is True

