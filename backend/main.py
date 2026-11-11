"""
main.py - FastAPI Application, WebSocket lifecycle, JIT Identity Binding,
Doctor Consent Handshake, Clinician Reasoning Inspector, and Interactive Chat Interface for Person 2.
"""

import os
import re
import time
import json
import uuid
import asyncio
from typing import Dict, Set, Optional, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from backend.state import ClinicalGraphState, DispatchedAction
from backend.database import (
    init_db,
    save_session_state,
    get_session_by_id,
    lock_session,
    bind_identity,
    has_identity_bound,
    get_identity_record,
    create_report_token,
    get_report_by_token,
    authorize_report_token,
    schedule_milestone,
    get_milestones_for_session,
    get_all_recent_sessions,
    get_full_session_telemetry
)
from backend.engine import clinical_engine


# In-memory WebSocket connection registry
active_connections: Dict[str, Set[WebSocket]] = {}
observer_connections: Dict[str, Set[WebSocket]] = {}


async def register_socket(session_id: str, websocket: WebSocket):
    if session_id not in active_connections:
        active_connections[session_id] = set()
    active_connections[session_id].add(websocket)


async def unregister_socket(session_id: str, websocket: WebSocket):
    if session_id in active_connections:
        active_connections[session_id].discard(websocket)
        if not active_connections[session_id]:
            del active_connections[session_id]


async def broadcast_to_session(session_id: str, message: dict):
    if session_id in active_connections:
        for ws in list(active_connections[session_id]):
            try:
                await ws.send_json(message)
            except Exception:
                pass


async def register_observer(session_id: str, websocket: WebSocket):
    if session_id not in observer_connections:
        observer_connections[session_id] = set()
    observer_connections[session_id].add(websocket)


async def unregister_observer(session_id: str, websocket: WebSocket):
    if session_id in observer_connections:
        observer_connections[session_id].discard(websocket)
        if not observer_connections[session_id]:
            del observer_connections[session_id]


async def broadcast_to_observers(session_id: str, message: dict):
    if session_id in observer_connections:
        for ws in list(observer_connections[session_id]):
            try:
                await ws.send_json(message)
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables in Supabase (or configured DB) on startup
    init_db()
    yield


app = FastAPI(
    title="Abu Dhabi DoH Jawda Clinical Voice Navigation Agent",
    description="Person 2: Clinical State Machine, Dynamic Gating, Zero-Knowledge Storage, JIT Enforcement, and Live Inspector",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -------------------------------------------------------------------------

class BindIdentityRequest(BaseModel):
    full_name: str = Field(..., min_length=2)
    emirates_id: str = Field(..., description="Emirates ID in 784-YYYY-XXXXXXX-Z format")
    insurance_carrier: str = Field(..., description="DAMAN_BASIC, DAMAN_ENHANCED, or THIQA")
    insurance_member_id: str = Field(..., min_length=3)
    phone: Optional[str] = None

    @field_validator("emirates_id")
    @classmethod
    def validate_emirates_id(cls, v: str) -> str:
        pattern = r"^784-[0-9]{4}-[0-9]{7}-[0-9]{1}$"
        if not re.match(pattern, v):
            raise ValueError("Emirates ID must strictly match format ^784-[0-9]{4}-[0-9]{7}-[0-9]{1}$")
        return v


class AuthorizeConsentRequest(BaseModel):
    access_token: str
    granted: bool


def _build_fhir_bundle(session_id: str, state: dict) -> dict:
    """Build HL7 FHIR R4 DiagnosticReport bundle."""
    return {
        "resourceType": "Bundle",
        "type": "document",
        "id": f"bundle-{session_id}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entry": [
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": f"dr-{session_id}",
                    "status": "final",
                    "code": {
                        "coding": [
                            {"system": "http://loinc.org", "code": "44249-1", "display": "PHQ-9 Depression Screener"},
                            {"system": "http://loinc.org", "code": "69737-5", "display": "GAD-7 Anxiety Screener"}
                        ]
                    },
                    "conclusion": f"Classified Triage Tier: {state.get('triage_tier')}, Recommended Modality: {state.get('care_modality')}",
                    "result": [
                        {
                            "observation": "PHQ-9 Total",
                            "score": state.get("phq9_total", 0),
                            "breakdown": state.get("phq9_slots", {})
                        },
                        {
                            "observation": "GAD-7 Total",
                            "score": state.get("gad7_total", 0),
                            "breakdown": state.get("gad7_slots", {})
                        },
                        {
                            "observation": "Somatic Findings (SNOMED CT)",
                            "findings": state.get("somatic_flags", [])
                        },
                        {
                            "observation": "Acoustic Biomarkers",
                            "arousal": state.get("vocal_arousal", 0.5),
                            "valence": state.get("vocal_valence", 0.5),
                            "affect_discrepancy": state.get("affect_discrepancy", False)
                        }
                    ]
                }
            }
        ]
    }


def _initialize_session_state(session_id: str, db_record: Optional[dict] = None) -> ClinicalGraphState:
    """Build initial or restored in-memory ClinicalGraphState."""
    if db_record:
        phq9 = json.loads(db_record.get("phq9_json", "{}"))
        gad7 = json.loads(db_record.get("gad7_json", "{}"))
        somatic = json.loads(db_record.get("somatic_json", "[]"))
        acoustic = json.loads(db_record.get("acoustic_json", "{}"))
        return {
            "session_id": session_id,
            "language": db_record.get("language", "ar-AE"),
            "active_mode": db_record.get("active_mode", "CONVERSATIONAL"),
            "messages": [],
            "vocal_arousal": acoustic.get("arousal", 0.5),
            "vocal_valence": acoustic.get("valence", 0.5),
            "affect_discrepancy": acoustic.get("affect_discrepancy", False),
            "phq9_slots": phq9,
            "phq9_total": sum(s["score"] for s in phq9.values()),
            "gad7_slots": gad7,
            "gad7_total": sum(s["score"] for s in gad7.values()),
            "cssrs_slots": {},
            "cssrs_screened": False,
            "cssrs_critical": bool(db_record.get("is_locked", 0)),
            "somatic_flags": somatic,
            "triage_tier": db_record.get("triage_tier", "TIER_1_SELF_CARE"),
            "care_modality": db_record.get("care_modality", "CONVERSATIONAL"),
            "reply_text": "",
            "ssml_tags": {
                "voice": "ar-AE-FatimaNeural",
                "style": "calm",
                "styledegree": "1.0",
                "rate": "-5%",
                "pitch": "0%",
                "pause_duration": "250ms"
            },
            "dispatched_action": None,
            "scheduled_reminders": [],
            "agentic_reasoning": None
        }
    return {
        "session_id": session_id,
        "language": "ar-AE",
        "active_mode": "CONVERSATIONAL",
        "messages": [],
        "vocal_arousal": 0.5,
        "vocal_valence": 0.5,
        "affect_discrepancy": False,
        "phq9_slots": {},
        "phq9_total": 0,
        "gad7_slots": {},
        "gad7_total": 0,
        "cssrs_slots": {},
        "cssrs_screened": False,
        "cssrs_critical": False,
        "somatic_flags": [],
        "triage_tier": "TIER_1_SELF_CARE",
        "care_modality": "CONVERSATIONAL",
        "reply_text": "",
        "ssml_tags": {
            "voice": "ar-AE-FatimaNeural",
            "style": "calm",
            "styledegree": "1.0",
            "rate": "-5%",
            "pitch": "0%",
            "pause_duration": "250ms"
        },
        "dispatched_action": None,
        "scheduled_reminders": [],
        "agentic_reasoning": None
    }


# -------------------------------------------------------------------------
# WEBSOCKET ENDPOINTS
# -------------------------------------------------------------------------

@app.websocket("/ws/triage/{session_id}")
async def websocket_triage(websocket: WebSocket, session_id: str):
    await websocket.accept()
    await register_socket(session_id, websocket)

    # Load or initialize state from DB
    db_record = get_session_by_id(session_id)
    state = _initialize_session_state(session_id, db_record)

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                packet = json.loads(raw_text)
            except Exception:
                await websocket.send_json({"type": "ERROR", "message": "Invalid JSON packet format"})
                continue

            packet_type = packet.get("type")

            # -------------------------------------------------------------
            # GATE 3A RE-ENTRY: User Interrupts Breathing Mode
            # -------------------------------------------------------------
            if packet_type == "USER_INTERRUPT_MODE":
                if state.get("active_mode") == "BREATHING_EXERCISE":
                    state["active_mode"] = "CONVERSATIONAL"
                    state["care_modality"] = "CONVERSATIONAL"
                    save_session_state(state)
                    is_ar = (state.get("language", "ar-AE") == "ar-AE")
                    resume_reply = (
                        "أهلاً بك مجدداً. أنا معك وجاهز لإكمال المحادثة متى ما أردت."
                        if is_ar else
                        "Welcome back. I am right here with you and ready to continue whenever you wish."
                    )
                    turn_data = {
                        "type": "AGENT_TURN",
                        "reply_text": resume_reply,
                        "ssml_tags": {
                            "voice": "ar-AE-FatimaNeural" if is_ar else "en-US-JennyMultilingualNeural",
                            "style": "calm",
                            "styledegree": "1.0",
                            "rate": "-5%",
                            "pitch": "0%",
                            "pause_duration": "250ms"
                        },
                        "dispatched_action": {
                            "action_type": "RESUME_CONVERSATION",
                            "payload": {"active_mode": "CONVERSATIONAL"}
                        },
                        "dashboard_telemetry": {
                            "phq9_total": state["phq9_total"],
                            "gad7_total": state["gad7_total"],
                            "somatic_count": len(state["somatic_flags"]),
                            "triage_tier": state["triage_tier"],
                            "active_mode": "CONVERSATIONAL"
                        },
                        "agentic_reasoning": {
                            "gate_3_reentry": {
                                "action": "USER_INTERRUPT_MODE",
                                "target_mode": "CONVERSATIONAL",
                                "rationale": "Bi-directional re-entry loop triggered. Smoothly returned patient to conversation without erasing cumulative slots."
                            }
                        }
                    }
                    await websocket.send_json(turn_data)
                    await broadcast_to_observers(session_id, {
                        "type": "OBSERVER_TURN_UPDATE",
                        "session_id": session_id,
                        "user_transcript": "[USER_INTERRUPT_MODE]",
                        "agent_reply": resume_reply,
                        "ssml_tags": turn_data["ssml_tags"],
                        "dispatched_action": turn_data["dispatched_action"],
                        "dashboard_telemetry": turn_data["dashboard_telemetry"],
                        "agentic_reasoning": turn_data["agentic_reasoning"]
                    })
                continue

            # -------------------------------------------------------------
            # STANDARD INBOUND TURN
            # -------------------------------------------------------------
            transcript = packet.get("transcript", "").strip()
            if not transcript:
                continue

            # Check if session is already emergency locked
            rec = get_session_by_id(session_id)
            if (rec and rec.get("is_locked")) or state.get("active_mode") == "EMERGENCY_LOCK":
                await websocket.send_json({
                    "type": "ERROR",
                    "error_code": "SESSION_LOCKED",
                    "message": "Session is locked under Hard Emergency Protocol. Immediate assistance is available via 800-SAKINA (800968) or Estijaba (8001717)."
                })
                continue

            # Check if triage decision was already finalized (Level 1, 2, or 3)
            prev_action = state.get("dispatched_action")
            prev_type = prev_action.get("action_type") if prev_action else None
            if prev_type in ("HARD_EMERGENCY_OVERRIDE", "TRIGGER_JIT_IDENTITY_GATE", "INITIATE_TELEHEALTH_HANDOFF", "SWITCH_UI_MODE"):
                is_ar = (state.get("language", "ar-AE") == "ar-AE")
                await websocket.send_json({
                    "type": "SESSION_FINALIZED",
                    "reply_text": "اكتمل التقييم السريري وتم اعتماد مسار الرعاية. تم إغلاق المحادثة." if is_ar else "Clinical triage assessment has been finalized. Chat is concluded for care pathway execution.",
                    "triage_tier": state.get("triage_tier"),
                    "dispatched_action": prev_action
                })
                continue

            # Ingest acoustic biomarkers & language
            arousal = float(packet.get("arousal", state.get("vocal_arousal", 0.5)))
            valence = float(packet.get("valence", state.get("vocal_valence", 0.5)))
            lang = packet.get("language", state.get("language", "ar-AE"))

            state["vocal_arousal"] = arousal
            state["vocal_valence"] = valence
            state["language"] = lang
            state["messages"].append({"role": "user", "content": transcript})

            # Run LangGraph Engine
            updated_state = clinical_engine.invoke(state)
            state.update(updated_state)

            # Gate 4 Conversational Anti-Bypass Check
            is_bound = has_identity_bound(session_id)
            t_tier = state.get("triage_tier")
            c_mod = state.get("care_modality")

            # Check if verbal bypass attempted or Level 2/3 requires identity
            verbal_bypass_attempt = any(
                phrase in transcript.lower()
                for phrase in ["just book", "book appointment", "book the appointment", "احجز لي", "احجز الموعد"]
            )

            if verbal_bypass_attempt and not is_bound:
                state["dispatched_action"] = {
                    "action_type": "TRIGGER_JIT_IDENTITY_GATE",
                    "payload": {
                        "status": "LOCKED_AWAITING_ID",
                        "enforcement": "IDENTITY_REQUIRED",
                        "facility": "Al Bateen Healthcare Center (AHS / SEHA)" if c_mod == "IN_CLINIC_VISIT" else "SEHA Virtual Consultation",
                        "message": "Emirates ID binding required before reservation can be confirmed."
                    }
                }
                is_ar = (state.get("language", "ar-AE") == "ar-AE")
                state["reply_text"] = (
                    "عشان نحجز لك الموعد ونربطه بملفك الطبي، يرجى استكمال بيانات بطاقة الهوية الإماراتية الظاهرة أمامك."
                    if is_ar else
                    "To hold this appointment and link your health record, please complete the Emirates ID details on your screen."
                )

            # Save state to DB
            if state.get("cssrs_critical") or state.get("active_mode") == "EMERGENCY_LOCK":
                lock_session(session_id)
                save_session_state(state, is_locked=True)
            else:
                save_session_state(state)

            # Build comprehensive turn payload
            turn_response = {
                "type": "AGENT_TURN",
                "reply_text": state["reply_text"],
                "ssml_tags": state["ssml_tags"],
                "dispatched_action": state["dispatched_action"],
                "dashboard_telemetry": {
                    "phq9_total": state["phq9_total"],
                    "gad7_total": state["gad7_total"],
                    "cssrs_count": len(state.get("cssrs_slots", {})),
                    "cssrs_critical": state.get("cssrs_critical", False),
                    "cssrs_screened": bool(state.get("cssrs_slots")) or state.get("cssrs_screened", False),
                    "somatic_count": len(state["somatic_flags"]),
                    "triage_tier": state["triage_tier"],
                    "active_mode": state["active_mode"]
                },
                "slots": {
                    "phq9_slots": state.get("phq9_slots", {}),
                    "gad7_slots": state.get("gad7_slots", {}),
                    "cssrs_slots": state.get("cssrs_slots", {}),
                    "somatic_flags": state.get("somatic_flags", []),
                    "cssrs_screened": bool(state.get("cssrs_slots")) or state.get("cssrs_screened", False),
                    "cssrs_critical": state.get("cssrs_critical", False)
                },
                "agentic_reasoning": state.get("agentic_reasoning")
            }

            # Send to patient WebSocket
            await websocket.send_json(turn_response)

            # Broadcast live to observer WebSockets
            await broadcast_to_observers(session_id, {
                "type": "OBSERVER_TURN_UPDATE",
                "session_id": session_id,
                "user_transcript": transcript,
                "agent_reply": state["reply_text"],
                "ssml_tags": state["ssml_tags"],
                "dispatched_action": state["dispatched_action"],
                "dashboard_telemetry": turn_response["dashboard_telemetry"],
                "slots": turn_response["slots"],
                "agentic_reasoning": state.get("agentic_reasoning")
            })

    except WebSocketDisconnect:
        await unregister_socket(session_id, websocket)
    except Exception:
        await unregister_socket(session_id, websocket)


@app.websocket("/ws/observer/{session_id}")
async def websocket_observer(websocket: WebSocket, session_id: str):
    """Real-time observer stream for clinicians inspecting agentic reasoning."""
    await websocket.accept()
    await register_observer(session_id, websocket)

    # Send current snapshot on connect
    telemetry = get_full_session_telemetry(session_id)
    await websocket.send_json({
        "type": "OBSERVER_CONNECTED",
        "session_id": session_id,
        "telemetry": telemetry
    })

    try:
        while True:
            # Keep socket open and listen for pings
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await unregister_observer(session_id, websocket)
    except Exception:
        await unregister_observer(session_id, websocket)


# -------------------------------------------------------------------------
# REST ENDPOINTS: CLINICIAN INSPECTOR & TELEMETRY
# -------------------------------------------------------------------------

@app.get("/api/sessions")
async def api_get_recent_sessions(limit: int = Query(25, ge=1, le=100)):
    """Retrieve recent triage sessions for clinician observer selector."""
    sessions = get_all_recent_sessions(limit)
    return {"sessions": sessions}


@app.get("/api/session/{session_id}/telemetry")
async def api_get_session_telemetry(session_id: str):
    """Retrieve full decoupled telemetry, identity vault status, and reports."""
    telemetry = get_full_session_telemetry(session_id)
    if not telemetry:
        raise HTTPException(status_code=404, detail="Session not found")
    return telemetry


# -------------------------------------------------------------------------
# REST ENDPOINTS: GATE 4 (JIT IDENTITY BINDING)
# -------------------------------------------------------------------------

@app.post("/api/session/{session_id}/bind-identity")
async def api_bind_identity(session_id: str, req: BindIdentityRequest):
    """
    Gate 4: Binds identity into decoupled zero-knowledge identity_vault table.
    Generates 6-character access token, creates encrypted FHIR brief, and schedules milestones.
    """
    # Verify session existence
    session = get_session_by_id(session_id)
    if not session:
        initial_state = _initialize_session_state(session_id)
        save_session_state(initial_state)
        session = get_session_by_id(session_id)

    # 1. Bind in identity vault (Zero-knowledge: hashed session id, masked EID, encrypted PII)
    vault_entry = bind_identity(
        session_id=session_id,
        full_name=req.full_name,
        emirates_id=req.emirates_id,
        insurance_carrier=req.insurance_carrier,
        insurance_member_id=req.insurance_member_id,
        phone=req.phone
    )

    # 2. Build and encrypt FHIR DiagnosticReport bundle
    state = _initialize_session_state(session_id, session)
    fhir_bundle = _build_fhir_bundle(session_id, state)
    access_token = create_report_token(session_id, fhir_bundle, ttl_seconds=3600)

    # 3. Schedule the 3 longitudinal milestones (Gate 6)
    now = time.time()
    schedule_milestone(
        session_id=session_id,
        touchpoint_type="PRE_VISIT_24H",
        target_timestamp=now + 86400,
        payload={"description": "Orientation Guide to reduce consultation anxiety"}
    )
    schedule_milestone(
        session_id=session_id,
        touchpoint_type="PRE_VISIT_1H",
        target_timestamp=now + 3600,
        payload={"description": "Fast-Track Check-In & Clinic GPS / WebRTC Link"}
    )
    schedule_milestone(
        session_id=session_id,
        touchpoint_type="JAWDA_30D",
        target_timestamp=now + (30 * 86400),
        payload={"description": "60-second voice check-in calculating symptom delta against DoH Jawda benchmark"}
    )

    # 4. Notify active WebSocket client and observers
    bound_event = {
        "type": "IDENTITY_BOUND_SUCCESS",
        "access_token": access_token,
        "masked_eid": vault_entry["masked_eid"],
        "appointment_status": "CONFIRMED"
    }
    await broadcast_to_session(session_id, bound_event)
    await broadcast_to_observers(session_id, bound_event)

    return {
        "status": "SUCCESS",
        "access_token": access_token,
        "masked_eid": vault_entry["masked_eid"],
        "appointment_status": "CONFIRMED"
    }


# -------------------------------------------------------------------------
# REST ENDPOINTS: GATE 5 (DOCTOR CONSENT HANDSHAKE & FHIR BRIEF)
# -------------------------------------------------------------------------

@app.get("/api/reports/request-access/{access_token}")
async def api_request_report_access(access_token: str):
    """
    Gate 5 Step 1: Doctor inputs token in Clinician Portal.
    Validates TTL and dispatches CONSENT_REQUEST_CHALLENGE to patient's active WebSocket.
    """
    report = get_report_by_token(access_token)
    if not report:
        raise HTTPException(status_code=404, detail="Clinical report token not found")

    now = time.time()
    expires_at = float(report["token_expires_at"])
    if now > expires_at:
        raise HTTPException(status_code=410, detail="Clinical report token has expired")

    session_id = report["session_id"]
    remaining_seconds = int(expires_at - now)

    # Push consent challenge over patient's WebSocket and observer stream
    challenge = {
        "type": "CONSENT_REQUEST_CHALLENGE",
        "access_token": access_token,
        "doctor": "Dr. Maryam",
        "facility": "AHS Al Bateen Healthcare Center"
    }
    await broadcast_to_session(session_id, challenge)
    await broadcast_to_observers(session_id, challenge)

    return {
        "status": "CONSENT_CHALLENGE_DISPATCHED",
        "expires_in_seconds": remaining_seconds
    }


@app.post("/api/reports/authorize")
async def api_authorize_report_access(req: AuthorizeConsentRequest):
    """
    Gate 5 Step 2: Patient authorizes or revokes access.
    Returns decrypted FHIR bundle if granted == True.
    """
    report = get_report_by_token(req.access_token)
    if not report:
        raise HTTPException(status_code=404, detail="Token not found")

    decrypted_fhir = authorize_report_token(req.access_token, req.granted)
    if not req.granted:
        return {"status": "REVOKED", "message": "Access denied by patient"}

    if decrypted_fhir is None:
        raise HTTPException(status_code=400, detail="Token expired or failed decryption")

    return {
        "status": "ACCESS_GRANTED",
        "fhir_diagnostic_report": decrypted_fhir
    }


# -------------------------------------------------------------------------
# CLINICIAN LIVE AGENTIC REASONING INSPECTOR UI (/observer)
# -------------------------------------------------------------------------

@app.get("/observer", response_class=HTMLResponse)
@app.get("/observer/{target_session_id}", response_class=HTMLResponse)
async def serve_observer_dashboard(target_session_id: Optional[str] = None):
    """Dedicated clinician dashboard for real-time agentic reasoning inspection."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DoH Jawda - Clinician Agentic Reasoning Inspector</title>
  <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --primary: #0A5C36;
      --primary-light: #E8F5E9;
      --accent: #D4AF37;
      --danger: #D32F2F;
      --danger-bg: #FFEBEE;
      --text: #1C2826;
      --bg: #F4F7F6;
      --surface: #FFFFFF;
      --border: #E0E7E5;
      --code-bg: #1E293B;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }}
    body {{ background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
    header {{ background: #063821; color: white; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid var(--accent); }}
    .header-left {{ display: flex; align-items: center; gap: 14px; }}
    .logo-badge {{ background: var(--accent); color: black; font-weight: 800; font-size: 11px; padding: 3px 8px; border-radius: 4px; }}
    .nav-btn {{ background: rgba(255,255,255,0.15); color: white; text-decoration: none; padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 600; }}
    .nav-btn:hover {{ background: rgba(255,255,255,0.25); }}

    /* Session selector bar */
    .session-bar {{ background: white; border-bottom: 1px solid var(--border); padding: 10px 24px; display: flex; gap: 16px; align-items: center; font-size: 13px; }}
    .session-bar select, .session-bar input {{ padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; outline: none; }}
    .btn-action {{ background: var(--primary); color: white; border: none; padding: 6px 14px; border-radius: 6px; font-weight: 600; cursor: pointer; }}
    .live-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #4CAF50; display: inline-block; animation: blink 1.5s infinite; }}

    /* Split layout */
    .layout {{ display: flex; flex: 1; height: calc(100vh - 105px); }}
    
    /* Left: Chat stream */
    .chat-col {{ flex: 0.9; background: white; border-right: 1px solid var(--border); display: flex; flex-direction: column; }}
    .col-header {{ padding: 12px 18px; border-bottom: 1px solid var(--border); background: #FAFDFB; font-weight: 700; font-size: 13px; color: #444; display: flex; justify-content: space-between; align-items: center; }}
    .chat-stream {{ flex: 1; padding: 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }}
    .turn-card {{ border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; font-size: 13px; line-height: 1.4; }}
    .turn-user {{ background: #F8FAFC; border-left: 4px solid #3B82F6; }}
    .turn-agent {{ background: #F0FDF4; border-left: 4px solid var(--primary); }}
    .turn-meta {{ font-size: 11px; color: #64748B; margin-bottom: 4px; font-weight: 600; display: flex; justify-content: space-between; }}

    /* Right: Clinical Reasoning Pipeline */
    .reasoning-col {{ flex: 1.4; background: var(--bg); overflow-y: auto; padding: 18px; display: flex; flex-direction: column; gap: 16px; }}
    .gate-card {{ background: white; border-radius: 10px; border: 1px solid var(--border); padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }}
    .gate-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .gate-num {{ font-size: 11px; font-weight: 800; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }}
    .gate-title {{ font-size: 14px; font-weight: 700; color: var(--primary); }}
    .status-badge {{ font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; }}
    .status-pass {{ background: #DCFCE7; color: #166534; }}
    .status-alert {{ background: #FEE2E2; color: #991B1B; }}
    .status-info {{ background: #EFF6FF; color: #1E40AF; }}

    .reasoning-text {{ font-size: 13px; color: #334155; line-height: 1.5; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px; border-radius: 6px; margin-top: 6px; }}
    
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }}
    .metric-pill {{ background: #F1F5F9; padding: 8px 12px; border-radius: 6px; font-size: 12px; }}
    .metric-pill b {{ color: var(--primary); }}

    pre.code-block {{ background: var(--code-bg); color: #94A3B8; padding: 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 11px; overflow-x: auto; margin-top: 8px; }}

    /* Signal Tracker & Question Indicators */
    .signal-tracker-box {{ background: #F8FAFC; border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-top: 10px; }}
    .signal-tracker-header {{ margin-bottom: 10px; }}
    .signal-title {{ display: flex; justify-content: space-between; font-size: 12px; font-weight: 700; color: #334155; margin-bottom: 6px; }}
    .progress-bar-bg {{ width: 100%; height: 8px; background: #E2E8F0; border-radius: 4px; overflow: hidden; }}
    .progress-bar-fill {{ height: 100%; background: linear-gradient(90deg, #10B981, #059669); width: 0%; transition: width 0.4s ease; }}
    
    .instrument-counters-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
    .inst-counter {{ background: white; border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; display: flex; justify-content: space-between; align-items: center; font-size: 11px; }}
    .inst-name {{ font-weight: 600; color: #475569; }}
    .inst-val {{ font-weight: 700; color: var(--primary); }}

    .question-tabs {{ display: flex; gap: 6px; margin-top: 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; overflow-x: auto; }}
    .q-tab {{ background: #F1F5F9; border: 1px solid var(--border); padding: 5px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; color: #475569; cursor: pointer; transition: 0.2s; white-space: nowrap; }}
    .q-tab:hover {{ background: #E2E8F0; }}
    .q-tab.active {{ background: var(--primary); color: white; border-color: var(--primary); }}

    .q-list {{ margin-top: 10px; display: flex; flex-direction: column; gap: 6px; max-height: 280px; overflow-y: auto; }}
    .q-row {{ background: white; border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-size: 12px; display: flex; flex-direction: column; gap: 4px; transition: 0.15s; }}
    .q-row.q-extracted {{ border-left: 4px solid #10B981; background: #FAFDFB; }}
    .q-row.q-pending {{ border-left: 4px solid #CBD5E1; opacity: 0.85; }}
    .q-row.q-alert {{ border-left: 4px solid var(--danger); background: #FFF5F5; }}
    
    .q-row-top {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
    .q-label {{ font-weight: 600; color: #1E293B; flex: 1; }}
    .q-code {{ font-size: 10px; color: #94A3B8; font-family: 'JetBrains Mono', monospace; margin-left: 6px; }}
    .q-badge {{ font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 4px; text-transform: uppercase; white-space: nowrap; }}
    .badge-extracted {{ background: #D1FAE5; color: #065F46; }}
    .badge-pending {{ background: #F1F5F9; color: #64748B; }}
    .badge-alert {{ background: #FEE2E2; color: #991B1B; }}
    .badge-safe {{ background: #E0F2FE; color: #0369A1; }}
    
    .q-evidence {{ font-size: 11px; color: #64748B; font-style: italic; background: rgba(0,0,0,0.03); padding: 3px 6px; border-radius: 4px; }}

    @keyframes blink {{
      0% {{ opacity: 1; }}
      50% {{ opacity: 0.3; }}
      100% {{ opacity: 1; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <h2 style="font-size: 16px; font-weight: 700;">Abu Dhabi Department of Health - Clinician Agentic Inspector</h2>
      <span class="logo-badge">JAWDA COMPLIANCE CORE</span>
    </div>
    <div style="display: flex; gap: 12px; align-items: center;">
      <a href="/chat" class="nav-btn" target="_blank">Open Patient Chat ↗</a>
      <span style="font-size: 12px; opacity: 0.85;"><span class="live-dot"></span> Live Telemetry</span>
    </div>
  </header>

  <div class="session-bar">
    <span>Active Session:</span>
    <select id="selSession" onchange="switchSession(this.value)">
      <option value="">Select or wait for sessions...</option>
    </select>
    <input type="text" id="txtCustomSession" placeholder="Or enter session_id" style="width: 220px;" value="{target_session_id or ''}">
    <button class="btn-action" onclick="loadCustomSession()">Connect & Inspect</button>
    <button class="btn-action" style="background: #475569;" onclick="refreshSessionList()">Refresh List</button>
  </div>

  <div class="layout">
    <!-- Left Chat Stream -->
    <div class="chat-col">
      <div class="col-header">
        <span>Patient & Agent Live Dialogue Stream</span>
        <span id="lblStreamStatus" style="font-size: 11px; color: #64748B;">Connecting...</span>
      </div>
      <div class="chat-stream" id="chatStream">
        <div style="text-align: center; color: #94A3B8; font-size: 12px; margin-top: 40px;">Select or connect to a session above to stream dialogue and internal agentic reasoning.</div>
      </div>
    </div>

    <!-- Right Reasoning Pipeline -->
    <div class="reasoning-col">
      <!-- Gate 0 -->
      <div class="gate-card">
        <div class="gate-header">
          <div>
            <div class="gate-num">Gate 0: Contextual Safety Circuit Breaker</div>
            <div class="gate-title">Columbia-Suicide Severity Rating Scale (C-SSRS)</div>
          </div>
          <span class="status-badge status-pass" id="badgeGate0">PASSED (SAFE)</span>
        </div>
        <div class="reasoning-text" id="descGate0">
          Evaluated latest statement for critical suicidal ideation, intent, or fatalistic overdose vs medication adherence.
        </div>
      </div>

      <!-- Gate 1 -->
      <div class="gate-card">
        <div class="gate-header">
          <div>
            <div class="gate-num">Gate 1: Psychometric Slot Extraction & Somatic Red Flags</div>
            <div class="gate-title">PHQ-9 (LOINC 44249-1) & GAD-7 (LOINC 69737-5) + SNOMED CT</div>
          </div>
          <span class="status-badge status-info" id="badgeGate1">ACTIVE EXTRACTION</span>
        </div>
        <div class="grid-2">
          <div class="metric-pill">PHQ-9 Depression: <b id="lblPhqTotal">0 / 27</b></div>
          <div class="metric-pill">GAD-7 Anxiety: <b id="lblGadTotal">0 / 21</b></div>
          <div class="metric-pill">Autonomic Arousal: <b id="lblArousal">0.50</b></div>
          <div class="metric-pill">Vocal Valence: <b id="lblValence">0.50</b></div>
        </div>
        <div class="reasoning-text" id="descGate1">
          Accumulates psychometric items monotonically without overwriting higher previous scores. Integrates acoustic biomarkers.
        </div>

        <!-- Master Signal Completion Tracker -->
        <div class="signal-tracker-box">
          <div class="signal-tracker-header">
            <div class="signal-title">
              <span>📊 Clinical Signal & Question Completion</span>
              <b id="lblOverallSignalRatio">0 / 22 Items Screened (0%)</b>
            </div>
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" id="barOverallCoverage" style="width: 0%;"></div>
            </div>
          </div>
          <div class="instrument-counters-grid">
            <div class="inst-counter">
              <span class="inst-name">PHQ-9 (Depression)</span>
              <b class="inst-val" id="lblPhqCount">0 / 9</b>
            </div>
            <div class="inst-counter">
              <span class="inst-name">GAD-7 (Anxiety)</span>
              <b class="inst-val" id="lblGadCount">0 / 7</b>
            </div>
            <div class="inst-counter">
              <span class="inst-name">C-SSRS (Safety)</span>
              <b class="inst-val" id="lblCssrsCount">0 / 6</b>
            </div>
            <div class="inst-counter">
              <span class="inst-name">Somatic Red Flags</span>
              <b class="inst-val" id="lblSomaticIndicator">0 Detected</b>
            </div>
          </div>
        </div>

        <!-- Question Tabs & Indicator Lists -->
        <div class="question-tabs">
          <button class="q-tab active" id="tabBtnPhq" onclick="switchInstTab('phq')">PHQ-9 Depression (9)</button>
          <button class="q-tab" id="tabBtnGad" onclick="switchInstTab('gad')">GAD-7 Anxiety (7)</button>
          <button class="q-tab" id="tabBtnCssrs" onclick="switchInstTab('cssrs')">C-SSRS Safety (6)</button>
          <button class="q-tab" id="tabBtnSomatic" onclick="switchInstTab('somatic')">Somatic Red Flags (4)</button>
          <button class="q-tab" id="tabBtnRaw" onclick="switchInstTab('raw')">Raw Payload</button>
        </div>

        <div id="tabContentPhq" class="q-list"></div>
        <div id="tabContentGad" class="q-list" style="display: none;"></div>
        <div id="tabContentCssrs" class="q-list" style="display: none;"></div>
        <div id="tabContentSomatic" class="q-list" style="display: none;"></div>
        <div id="tabContentRaw" style="display: none;">
          <pre class="code-block" id="jsonSlots">{{ "phq9_slots": {{}}, "gad7_slots": {{}}, "somatic_flags": [] }}</pre>
        </div>
      </div>

      <!-- Gate 2 -->
      <div class="gate-card">
        <div class="gate-header">
          <div>
            <div class="gate-num">Gate 2: Stepped-Care Classification Engine</div>
            <div class="gate-title">Abu Dhabi Severity Truth Table Matching</div>
          </div>
          <span class="status-badge status-pass" id="badgeGate2">TIER_1_SELF_CARE</span>
        </div>
        <div class="reasoning-text" id="descGate2">
          Calculates triage tier from cumulative PHQ-9, GAD-7, and somatic rule-out flags.
        </div>
      </div>

      <!-- Gate 3 -->
      <div class="gate-card">
        <div class="gate-header">
          <div>
            <div class="gate-num">Gate 3: Care Modality & UI Mode Dispatcher</div>
            <div class="gate-title">Grounding Canvas / Virtual Telehealth / Clinic Outpatient</div>
          </div>
          <span class="status-badge status-info" id="badgeGate3">CONVERSATIONAL</span>
        </div>
        <div class="reasoning-text" id="descGate3">
          Assigned Care Modality: CONVERSATIONAL. Mode switches only occur when situational distress or clinical appointments are indicated.
        </div>
      </div>

      <!-- Gate 4 & 5 -->
      <div class="gate-card">
        <div class="gate-header">
          <div>
            <div class="gate-num">Gate 4 & 5: Decoupled Zero-Knowledge Vault & Consent Handshake</div>
            <div class="gate-title">SHA-256 Vault Linkage, Ephemeral Access Token & FHIR R4 Bundle</div>
          </div>
          <span class="status-badge status-info" id="badgeGate4">GHOST ANONYMOUS</span>
        </div>
        <div class="grid-2">
          <div class="metric-pill">Masked EID: <b id="lblMaskedEid">None</b></div>
          <div class="metric-pill">Access Token: <b id="lblAccessToken">None</b></div>
        </div>
        <div class="reasoning-text" id="descGate4">
          PII is AES-256-GCM encrypted and linked solely via SHA-256(session_id). Plaintext personal details never enter clinical tables.
        </div>
      </div>

      <!-- Gate 6 -->
      <div class="gate-card">
        <div class="gate-header">
          <div>
            <div class="gate-num">Gate 6: Longitudinal Jawda Audit Milestones</div>
            <div class="gate-title">DoH Jawda Mental Health KPI Benchmark (ΔSymptom ≥ 50.0%)</div>
          </div>
          <span class="status-badge status-info" id="badgeGate6">PENDING</span>
        </div>
        <div class="reasoning-text">
          Registers 3 automated touchpoints on booking finalization: Day -1 (Orientation), 1h (Fast-Track GPS/Link), and Day +30 (Jawda Symptom Delta Audit).
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentSessionId = "{target_session_id or ''}";
    let obsWs = null;

    const PHQ9_QUESTIONS = [
      {{ id: "item1", code: "LOINC 44250-9", name: "1. Little interest or pleasure (Anhedonia)", keys: ["item1", "anhedonia", "pleasure", "interest"] }},
      {{ id: "item2", code: "LOINC 44255-8", name: "2. Feeling down, depressed, or hopeless", keys: ["item2", "depressed", "hopeless", "down"] }},
      {{ id: "item3", code: "LOINC 44259-0", name: "3. Trouble falling/staying asleep, or sleeping too much", keys: ["item3", "sleep", "insomnia"] }},
      {{ id: "item4", code: "LOINC 44254-1", name: "4. Feeling tired or having little energy", keys: ["item4", "fatigue", "tired", "energy"] }},
      {{ id: "item5", code: "LOINC 44251-7", name: "5. Poor appetite or overeating", keys: ["item5", "appetite", "eating", "eat"] }},
      {{ id: "item6", code: "LOINC 44258-2", name: "6. Feeling bad about yourself / failure", keys: ["item6", "guilt", "failure", "bad about"] }},
      {{ id: "item7", code: "LOINC 44252-5", name: "7. Trouble concentrating on things", keys: ["item7", "concentration", "concentrat"] }},
      {{ id: "item8", code: "LOINC 44253-3", name: "8. Moving/speaking slowly or fidgety/restless", keys: ["item8", "psychomotor", "slow", "fidgety", "restless"] }},
      {{ id: "item9", code: "LOINC 44260-8", name: "9. Thoughts of hurting yourself or better off dead", keys: ["item9", "suicide", "better off dead", "hurting yourself"] }}
    ];

    const GAD7_QUESTIONS = [
      {{ id: "item1", code: "LOINC 69725-0", name: "1. Feeling nervous, anxious, or on edge", keys: ["item1", "nervous", "anxious", "edge"] }},
      {{ id: "item2", code: "LOINC 68509-9", name: "2. Not being able to stop or control worrying", keys: ["item2", "uncontrolled", "stop worrying", "control worrying"] }},
      {{ id: "item3", code: "LOINC 69733-4", name: "3. Worrying too much about different things", keys: ["item3", "excessive", "too much about", "different things"] }},
      {{ id: "item4", code: "LOINC 69734-2", name: "4. Trouble relaxing", keys: ["item4", "relax", "trouble relaxing"] }},
      {{ id: "item5", code: "LOINC 69735-9", name: "5. Being so restless that it is hard to sit still", keys: ["item5", "restless", "sit still", "restlessness"] }},
      {{ id: "item6", code: "LOINC 69736-7", name: "6. Becoming easily annoyed or irritable", keys: ["item6", "irritab", "annoyed", "short-tempered"] }},
      {{ id: "item7", code: "LOINC 69737-5", name: "7. Feeling afraid as if something awful might happen", keys: ["item7", "dread", "awful", "afraid", "panic"] }}
    ];

    const CSSRS_QUESTIONS = [
      {{ id: "q1", name: "Q1. Wish to be Dead (Passive Ideation)", keys: ["q1", "wish_dead", "passive"], desc: "Wished you were dead or wished you could sleep and not wake up" }},
      {{ id: "q2", name: "Q2. Non-Specific Active Suicidal Thoughts", keys: ["q2", "active", "thoughts"], desc: "Actual general thoughts of killing yourself" }},
      {{ id: "q3", name: "Q3. Active Suicidal Ideation with Methods", keys: ["q3", "method", "ways"], desc: "Thinking about how you might do it (no intent)" }},
      {{ id: "q4", name: "Q4. Active Ideation with Some Intent", keys: ["q4", "intent"], desc: "Having thoughts with intent to act (CRITICAL)" }},
      {{ id: "q5", name: "Q5. Active Ideation with Specific Plan & Intent", keys: ["q5", "plan"], desc: "Specific plan worked out with intent (CRITICAL)" }},
      {{ id: "q6", name: "Q6. Suicidal Behavior / Preparatory Acts", keys: ["q6", "behavior", "preparatory", "attempt"], desc: "Preparatory steps or self-harm attempts (CRITICAL)" }}
    ];

    const SOMATIC_QUESTIONS = [
      {{ id: "271795006", name: "Precordial Chest Tightness / Pain", snomed: "SNOMED 271795006", ruleOut: true }},
      {{ id: "80313002", name: "Palpitations / Tachycardia", snomed: "SNOMED 80313002", ruleOut: true }},
      {{ id: "282291009", name: "Dyspnea / Shortness of Breath", snomed: "SNOMED 282291009", ruleOut: true }},
      {{ id: "398057008", name: "Tension Headache", snomed: "SNOMED 398057008", ruleOut: false }}
    ];

    function switchInstTab(tabKey) {{
      const tabs = ["phq", "gad", "cssrs", "somatic", "raw"];
      tabs.forEach(t => {{
        const btn = document.getElementById("tabBtn" + t.charAt(0).toUpperCase() + t.slice(1));
        const pane = document.getElementById("tabContent" + t.charAt(0).toUpperCase() + t.slice(1));
        if (btn && pane) {{
          if (t === tabKey) {{
            btn.classList.add("active");
            pane.style.display = (t === "raw" ? "block" : "flex");
          }} else {{
            btn.classList.remove("active");
            pane.style.display = "none";
          }}
        }}
      }});
    }}

    function findSlotMatch(slotsObj, keyPatterns) {{
      if (!slotsObj || typeof slotsObj !== "object") return null;
      for (const k of Object.keys(slotsObj)) {{
        const lk = k.toLowerCase();
        for (const pat of keyPatterns) {{
          if (lk.includes(pat)) return slotsObj[k];
        }}
      }}
      return null;
    }}

    function renderQuestionIndicators(phq, gad, cssrs, somatic, cssrsScreened, cssrsCritical) {{
      let phqScreenedCount = 0;
      let gadScreenedCount = 0;
      let cssrsScreenedCount = 0;
      let somaticDetectedCount = 0;

      // 1. Render PHQ-9
      const phqContainer = document.getElementById("tabContentPhq");
      if (phqContainer) {{
        phqContainer.innerHTML = "";
        PHQ9_QUESTIONS.forEach(q => {{
          const match = findSlotMatch(phq, q.keys);
          const score = (match && typeof match.score === "number") ? match.score : (match && typeof match === "number" ? match : 0);
          const evidence = (match && match.evidence) ? match.evidence : "";
          const isExtracted = score > 0;
          if (isExtracted) phqScreenedCount++;

          const row = document.createElement("div");
          row.className = `q-row ${{isExtracted ? 'q-extracted' : 'q-pending'}}`;
          
          let scoreText = "0 / Pending";
          let badgeClass = "badge-pending";
          if (score === 1) {{ scoreText = "Score: 1 (Several days)"; badgeClass = "badge-extracted"; }}
          else if (score === 2) {{ scoreText = "Score: 2 (> half days)"; badgeClass = "badge-extracted"; }}
          else if (score >= 3) {{ scoreText = "Score: 3 (Nearly every day)"; badgeClass = "badge-alert"; }}

          row.innerHTML = `
            <div class="q-row-top">
              <span class="q-label">${{q.name}} <span class="q-code">${{q.code}}</span></span>
              <span class="q-badge ${{badgeClass}}">${{scoreText}}</span>
            </div>
            ${{evidence ? `<div class="q-evidence">Evidence: "${{evidence}}"</div>` : ''}}
          `;
          phqContainer.appendChild(row);
        }});
      }}

      // 2. Render GAD-7
      const gadContainer = document.getElementById("tabContentGad");
      if (gadContainer) {{
        gadContainer.innerHTML = "";
        GAD7_QUESTIONS.forEach(q => {{
          const match = findSlotMatch(gad, q.keys);
          const score = (match && typeof match.score === "number") ? match.score : (match && typeof match === "number" ? match : 0);
          const evidence = (match && match.evidence) ? match.evidence : "";
          const isExtracted = score > 0;
          if (isExtracted) gadScreenedCount++;

          const row = document.createElement("div");
          row.className = `q-row ${{isExtracted ? 'q-extracted' : 'q-pending'}}`;
          
          let scoreText = "0 / Pending";
          let badgeClass = "badge-pending";
          if (score === 1) {{ scoreText = "Score: 1 (Several days)"; badgeClass = "badge-extracted"; }}
          else if (score === 2) {{ scoreText = "Score: 2 (> half days)"; badgeClass = "badge-extracted"; }}
          else if (score >= 3) {{ scoreText = "Score: 3 (Nearly every day)"; badgeClass = "badge-alert"; }}

          row.innerHTML = `
            <div class="q-row-top">
              <span class="q-label">${{q.name}} <span class="q-code">${{q.code}}</span></span>
              <span class="q-badge ${{badgeClass}}">${{scoreText}}</span>
            </div>
            ${{evidence ? `<div class="q-evidence">Evidence: "${{evidence}}"</div>` : ''}}
          `;
          gadContainer.appendChild(row);
        }});
      }}

      // 3. Render C-SSRS
      const cssrsContainer = document.getElementById("tabContentCssrs");
      if (cssrsContainer) {{
        cssrsContainer.innerHTML = "";
        CSSRS_QUESTIONS.forEach(q => {{
          const match = findSlotMatch(cssrs, q.keys);
          let isPositive = false;
          let isScreenedSafe = false;
          let evidence = "";

          if (match) {{
            evidence = match.evidence || "";
            if (match.score && match.score > 0) isPositive = true;
          }} else if (cssrsScreened && !cssrsCritical && (q.id === "q1" || q.id === "q2")) {{
            isScreenedSafe = true;
          }}

          if (isPositive || isScreenedSafe) cssrsScreenedCount++;

          const row = document.createElement("div");
          let badgeText = "Unscreened";
          let badgeClass = "badge-pending";
          let rowClass = "q-pending";

          if (isPositive) {{
            badgeText = "CRITICAL / POSITIVE";
            badgeClass = "badge-alert";
            rowClass = "q-alert";
          }} else if (isScreenedSafe) {{
            badgeText = "NEGATIVE (SAFE)";
            badgeClass = "badge-safe";
            rowClass = "q-extracted";
          }}

          row.className = `q-row ${{rowClass}}`;
          row.innerHTML = `
            <div class="q-row-top">
              <span class="q-label">${{q.name}}</span>
              <span class="q-badge ${{badgeClass}}">${{badgeText}}</span>
            </div>
            <div style="font-size: 11px; color: #64748B;">${{q.desc}}</div>
            ${{evidence ? `<div class="q-evidence">Evidence: "${{evidence}}"</div>` : ''}}
          `;
          cssrsContainer.appendChild(row);
        }});
      }}

      // 4. Render Somatic
      const somaticContainer = document.getElementById("tabContentSomatic");
      if (somaticContainer) {{
        somaticContainer.innerHTML = "";
        const somaticArray = Array.isArray(somatic) ? somatic : [];
        SOMATIC_QUESTIONS.forEach(q => {{
          const match = somaticArray.find(s => String(s.snomed_ct || "") === q.id);
          const isDetected = !!match;
          if (isDetected) somaticDetectedCount++;

          const row = document.createElement("div");
          let badgeText = "Ruled Out / Negative";
          let badgeClass = "badge-safe";
          let rowClass = "q-extracted";

          if (isDetected) {{
            badgeText = q.ruleOut ? "DETECTED - RULE OUT CARDIAC" : "DETECTED (BENIGN)";
            badgeClass = q.ruleOut ? "badge-alert" : "badge-extracted";
            rowClass = q.ruleOut ? "q-alert" : "q-extracted";
          }}

          row.className = `q-row ${{rowClass}}`;
          row.innerHTML = `
            <div class="q-row-top">
              <span class="q-label">${{q.name}} <span class="q-code">${{q.snomed}}</span></span>
              <span class="q-badge ${{badgeClass}}">${{badgeText}}</span>
            </div>
            <div style="font-size: 11px; color: #64748B;">Protocol: ${{q.ruleOut ? 'Mandatory physical physician evaluation if present' : 'Supportive care'}}</div>
          `;
          somaticContainer.appendChild(row);
        }});
      }}

      // 5. Update Counters & Progress Bar
      const totalScreened = phqScreenedCount + gadScreenedCount + (cssrsScreenedCount > 0 ? Math.max(cssrsScreenedCount, 1) : 0);
      const totalSignals = 22; // 9 PHQ + 7 GAD + 6 CSSRS
      const percent = Math.min(100, Math.round((totalScreened / totalSignals) * 100));

      const lblRatio = document.getElementById("lblOverallSignalRatio");
      if (lblRatio) lblRatio.innerText = `${{totalScreened}} / ${{totalSignals}} Items Screened (${{percent}}%)`;

      const barFill = document.getElementById("barOverallCoverage");
      if (barFill) barFill.style.width = `${{percent}}%`;

      const phqLbl = document.getElementById("lblPhqCount");
      if (phqLbl) phqLbl.innerText = `${{phqScreenedCount}} / 9`;

      const gadLbl = document.getElementById("lblGadCount");
      if (gadLbl) gadLbl.innerText = `${{gadScreenedCount}} / 7`;

      const cssrsLbl = document.getElementById("lblCssrsCount");
      if (cssrsLbl) cssrsLbl.innerText = `${{cssrsScreenedCount}} / 6`;

      const somaticLbl = document.getElementById("lblSomaticIndicator");
      if (somaticLbl) somaticLbl.innerText = `${{somaticDetectedCount}} Detected`;
    }}

    async function refreshSessionList() {{
      try {{
        const res = await fetch("/api/sessions?limit=25");
        const data = await res.json();
        const sel = document.getElementById("selSession");
        sel.innerHTML = '<option value="">Select a session...</option>';
        if (data.sessions) {{
          data.sessions.forEach(s => {{
            const opt = document.createElement("option");
            opt.value = s.session_id;
            const dt = new Date(s.created_at * 1000).toLocaleTimeString();
            opt.innerText = `${{s.session_id}} (${{s.triage_tier}} | ${{s.care_modality}} - ${{dt}})`;
            sel.appendChild(opt);
          }});
        }}
        if (currentSessionId) {{
          sel.value = currentSessionId;
        }}
      }} catch (err) {{
        console.error("Error fetching sessions:", err);
      }}
    }}

    function switchSession(sessId) {{
      if (!sessId) return;
      currentSessionId = sessId;
      document.getElementById("txtCustomSession").value = sessId;
      connectObserverWs(sessId);
      loadSessionTelemetry(sessId);
    }}

    function loadCustomSession() {{
      const val = document.getElementById("txtCustomSession").value.trim();
      if (val) {{
        switchSession(val);
      }}
    }}

    async function loadSessionTelemetry(sessId) {{
      try {{
        const res = await fetch(`/api/session/${{sessId}}/telemetry`);
        if (!res.ok) return;
        const data = await res.json();
        renderSnapshot(data);
      }} catch (err) {{
        console.error("Error loading telemetry:", err);
      }}
    }}

    function renderSnapshot(t) {{
      if (!t || !t.session) return;
      const s = t.session;
      const phq = JSON.parse(s.phq9_json || "{{}}");
      const gad = JSON.parse(s.gad7_json || "{{}}");
      const somatic = JSON.parse(s.somatic_json || "[]");
      const cssrs = JSON.parse(s.cssrs_json || "{{}}");
      const acoustic = JSON.parse(s.acoustic_json || "{{}}");

      const phqTotal = Object.values(phq).reduce((sum, item) => sum + (item.score || 0), 0);
      const gadTotal = Object.values(gad).reduce((sum, item) => sum + (item.score || 0), 0);

      document.getElementById("lblPhqTotal").innerText = `${{phqTotal}} / 27`;
      document.getElementById("lblGadTotal").innerText = `${{gadTotal}} / 21`;
      document.getElementById("lblArousal").innerText = (acoustic.arousal || 0.5).toFixed(2);
      document.getElementById("lblValence").innerText = (acoustic.valence || 0.5).toFixed(2);

      const b2 = document.getElementById("badgeGate2");
      b2.innerText = s.triage_tier;
      b2.className = `status-badge ${{s.triage_tier === 'TIER_EMERGENCY' ? 'status-alert' : 'status-pass'}}`;

      const b3 = document.getElementById("badgeGate3");
      b3.innerText = s.care_modality;

      document.getElementById("jsonSlots").innerText = JSON.stringify({{
        phq9_slots: phq,
        gad7_slots: gad,
        cssrs_slots: cssrs,
        somatic_flags: somatic
      }}, null, 2);

      renderQuestionIndicators(
        phq,
        gad,
        cssrs,
        somatic,
        s.is_locked ? true : (Object.keys(cssrs).length > 0),
        s.triage_tier === "TIER_EMERGENCY"
      );

      if (t.identity_vault && t.identity_vault.is_bound) {{
        document.getElementById("badgeGate4").innerText = "BOUND & ENCRYPTED";
        document.getElementById("badgeGate4").className = "status-badge status-pass";
        document.getElementById("lblMaskedEid").innerText = t.identity_vault.masked_eid;
      }} else {{
        document.getElementById("badgeGate4").innerText = "ANONYMOUS GHOST";
        document.getElementById("badgeGate4").className = "status-badge status-info";
        document.getElementById("lblMaskedEid").innerText = "None (anon_ghost)";
      }}

      if (t.clinical_reports && t.clinical_reports.length > 0) {{
        const r = t.clinical_reports[0];
        document.getElementById("lblAccessToken").innerText = `${{r.access_token}} (${{r.consent_status}})`;
      }}
    }}

    function connectObserverWs(sessId) {{
      if (obsWs) {{
        obsWs.close();
      }}
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      obsWs = new WebSocket(`${{protocol}}//${{window.location.host}}/ws/observer/${{sessId}}`);

      obsWs.onopen = () => {{
        document.getElementById("lblStreamStatus").innerText = `Connected: ${{sessId}}`;
      }};

      obsWs.onmessage = (event) => {{
        const data = JSON.parse(event.data);
        if (data.type === "OBSERVER_CONNECTED") {{
          renderSnapshot(data.telemetry);
        }} else if (data.type === "OBSERVER_TURN_UPDATE") {{
          handleLiveTurn(data);
        }} else if (data.type === "IDENTITY_BOUND_SUCCESS") {{
          document.getElementById("badgeGate4").innerText = "BOUND & ENCRYPTED";
          document.getElementById("badgeGate4").className = "status-badge status-pass";
          document.getElementById("lblMaskedEid").innerText = data.masked_eid;
          document.getElementById("lblAccessToken").innerText = data.access_token;
        }}
      }};

      obsWs.onclose = () => {{
        document.getElementById("lblStreamStatus").innerText = "Disconnected";
      }};
    }}

    function handleLiveTurn(turn) {{
      const stream = document.getElementById("chatStream");

      // Append user turn
      if (turn.user_transcript) {{
        const uDiv = document.createElement("div");
        uDiv.className = "turn-card turn-user";
        uDiv.innerHTML = `<div class="turn-meta"><span>Patient</span><span>Arousal: ${{turn.agentic_reasoning?.gate_1_extraction?.acoustic_biomarkers?.arousal || 0.5}}</span></div><div>${{turn.user_transcript}}</div>`;
        stream.appendChild(uDiv);
      }}

      // Append agent turn
      if (turn.agent_reply) {{
        const aDiv = document.createElement("div");
        aDiv.className = "turn-card turn-agent";
        aDiv.innerHTML = `<div class="turn-meta"><span>Clinician Agent</span><span>Voice: ${{turn.ssml_tags?.voice || 'ar-AE-FatimaNeural'}}</span></div><div>${{turn.agent_reply}}</div>`;
        stream.appendChild(aDiv);
      }}

      stream.scrollTop = stream.scrollHeight;

      // Update Reasoning panels
      const r = turn.agentic_reasoning;
      if (r) {{
        if (r.gate_0_safety) {{
          const g0 = r.gate_0_safety;
          const b0 = document.getElementById("badgeGate0");
          b0.innerText = g0.status;
          b0.className = `status-badge ${{g0.status === 'CRITICAL_TRIGGERED' ? 'status-alert' : 'status-pass'}}`;
          document.getElementById("descGate0").innerText = `Evaluated: "${{g0.evaluated_utterance}}" -> ${{g0.rationale}}`;
        }}

        if (r.gate_1_extraction) {{
          const g1 = r.gate_1_extraction;
          document.getElementById("lblPhqTotal").innerText = `${{g1.phq9_total}} / 27`;
          document.getElementById("lblGadTotal").innerText = `${{g1.gad7_total}} / 21`;
          document.getElementById("lblArousal").innerText = g1.acoustic_biomarkers.arousal.toFixed(2);
          document.getElementById("lblValence").innerText = g1.acoustic_biomarkers.valence.toFixed(2);
          document.getElementById("descGate1").innerText = g1.rationale;
          document.getElementById("jsonSlots").innerText = JSON.stringify({{
            phq9_slots: g1.phq9_slots,
            gad7_slots: g1.gad7_slots,
            cssrs_slots: g1.cssrs_slots,
            somatic_flags: g1.somatic_symptoms
          }}, null, 2);
        }}

        if (turn.slots) {{
          renderQuestionIndicators(
            turn.slots.phq9_slots || {{}},
            turn.slots.gad7_slots || {{}},
            turn.slots.cssrs_slots || {{}},
            turn.slots.somatic_flags || [],
            turn.slots.cssrs_screened || false,
            turn.slots.cssrs_critical || false
          );
        }} else if (r && r.gate_1_extraction) {{
          const g1 = r.gate_1_extraction;
          renderQuestionIndicators(
            g1.phq9_slots || {{}},
            g1.gad7_slots || {{}},
            g1.cssrs_slots || {{}},
            g1.somatic_symptoms || [],
            !!(g1.cssrs_slots && Object.keys(g1.cssrs_slots).length > 0),
            r.gate_0_safety?.status === "CRITICAL_TRIGGERED"
          );
        }}

        if (r.gate_2_classification) {{
          const g2 = r.gate_2_classification;
          const b2 = document.getElementById("badgeGate2");
          b2.innerText = g2.triage_tier;
          b2.className = `status-badge ${{g2.triage_tier === 'TIER_EMERGENCY' ? 'status-alert' : 'status-pass'}}`;
          document.getElementById("descGate2").innerText = g2.rule_matched;

          const b3 = document.getElementById("badgeGate3");
          b3.innerText = g2.care_modality;
        }}

        if (r.gate_3_routing) {{
          const g3 = r.gate_3_routing;
          document.getElementById("descGate3").innerText = `${{g3.action_type}}: ${{g3.rationale}}`;
        }}
      }}
    }}

    window.onload = () => {{
      renderQuestionIndicators({{}}, {{}}, {{}}, [], false, false);
      refreshSessionList();
      if (currentSessionId) {{
        switchSession(currentSessionId);
      }}
    }};
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


# -------------------------------------------------------------------------
# INTERACTIVE CHAT & CLINICAL DASHBOARD TESTING UI (/chat, /)
# -------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
async def serve_chat_ui():
    """Interactive testing chat interface for the DoH Jawda clinical state machine with Language Switching."""
    html_content = """<!DOCTYPE html>
<html lang="ar" dir="ltr" id="htmlRoot">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>دائرة الصحة أبوظبي - المساعد الصوتي السريري الذكي</title>
  <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #0A5C36;
      --primary-light: #E8F5E9;
      --accent: #D4AF37;
      --danger: #D32F2F;
      --danger-bg: #FFEBEE;
      --text: #1C2826;
      --bg: #F4F7F6;
      --surface: #FFFFFF;
      --border: #E0E7E5;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Tajawal', 'Inter', sans-serif; }
    body { background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
    header { background: var(--primary); color: white; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 8px rgba(0,0,0,0.12); direction: ltr !important; }
    .header-title { display: flex; align-items: center; gap: 12px; }
    .header-badge { background: var(--accent); color: #000; font-size: 11px; font-weight: 800; padding: 3px 8px; border-radius: 12px; letter-spacing: 0.5px; }
    
    .header-actions { display: flex; gap: 12px; align-items: center; }
    .lang-switcher { background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.4); border-radius: 20px; display: flex; padding: 2px; }
    .lang-btn { background: transparent; border: none; color: white; padding: 4px 12px; font-size: 12px; font-weight: 700; border-radius: 16px; cursor: pointer; transition: 0.2s; }
    .lang-btn.active { background: white; color: var(--primary); }
    .btn-inspector { background: var(--accent); color: #000; text-decoration: none; padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 700; display: flex; align-items: center; gap: 6px; }
    .btn-inspector:hover { opacity: 0.9; }

    /* Fix mirror inversion: Lock layout direction to LTR so columns never invert */
    .layout { display: flex; flex: 1; height: calc(100vh - 56px); direction: ltr !important; }
    
    /* Left / Chat Area */
    .chat-container { flex: 1.3; display: flex; flex-direction: column; background: var(--surface); border-right: 1px solid var(--border); position: relative; }
    .messages-area { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
    .message { max-width: 80%; padding: 12px 16px; border-radius: 14px; font-size: 14px; line-height: 1.6; }
    .message.user { align-self: flex-end; background: var(--primary); color: white; border-bottom-right-radius: 2px; }
    .message.agent { align-self: flex-start; background: var(--primary-light); color: var(--text); border-bottom-left-radius: 2px; border: 1px solid #C8E6C9; }
    
    /* Arabic text alignment without layout inversion */
    html[lang="ar"] .message { direction: rtl; text-align: right; font-family: 'Tajawal', sans-serif; }
    html[lang="ar"] .input-row input { direction: rtl; text-align: right; font-family: 'Tajawal', sans-serif; }
    html[lang="ar"] .card { text-align: right; }
    html[lang="ar"] .card-title { direction: rtl; }
    html[lang="ar"] .modal-box { text-align: right; direction: rtl; }
    
    .message.agent .prosody { font-size: 11px; opacity: 0.75; margin-top: 6px; font-family: monospace; direction: ltr; text-align: left; }
    
    /* Controls Footer */
    .controls-footer { padding: 14px 20px; background: white; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 10px; }
    .biomarker-bar { display: flex; gap: 16px; align-items: center; font-size: 12px; background: #FAFCFB; padding: 8px 12px; border-radius: 8px; border: 1px solid var(--border); }
    .slider-group { display: flex; align-items: center; gap: 8px; flex: 1; }
    .slider-group input { flex: 1; accent-color: var(--primary); }
    .presets { display: flex; gap: 6px; }
    .btn-preset { padding: 3px 8px; font-size: 11px; border: 1px solid var(--border); background: white; border-radius: 6px; cursor: pointer; }
    .btn-preset:hover { background: var(--primary-light); }
    .input-row { display: flex; gap: 10px; }
    .input-row input { flex: 1; padding: 12px 16px; border: 1px solid var(--border); border-radius: 8px; font-size: 14px; outline: none; }
    .input-row input:focus { border-color: var(--primary); }
    .btn-send { background: var(--primary); color: white; border: none; padding: 0 24px; border-radius: 8px; font-weight: 700; cursor: pointer; font-size: 14px; }
    .btn-send:hover { opacity: 0.9; }

    /* Right Telemetry Sidebar */
    .telemetry-sidebar { flex: 0.9; background: var(--bg); padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; border-right: 1px solid var(--border); }
    .card { background: white; border-radius: 12px; padding: 16px; border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.03); }
    .card-title { font-size: 13px; font-weight: 700; color: #555; text-transform: uppercase; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
    .telemetry-badge { padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; }
    .badge-tier-1 { background: #E8F5E9; color: #2E7D32; }
    .badge-tier-2 { background: #FFF3E0; color: #E65100; }
    .badge-tier-3 { background: #FFEBEE; color: #C62828; }
    .badge-emergency { background: #B71C1C; color: white; animation: pulse 1.5s infinite; }
    
    .score-meter { display: flex; gap: 12px; margin-bottom: 12px; }
    .score-box { flex: 1; background: #F8FAF9; padding: 10px; border-radius: 8px; border: 1px solid var(--border); text-align: center; }
    .score-num { font-size: 22px; font-weight: 800; color: var(--primary); }
    .score-lbl { font-size: 11px; color: #666; margin-top: 2px; }

    /* Modals & Overlays */
    .overlay { display: none; position: absolute; inset: 0; background: rgba(0,0,0,0.65); z-index: 100; justify-content: center; align-items: center; backdrop-filter: blur(4px); }
    .overlay.active { display: flex; }
    
    /* 4-7-8 Breathing Circle */
    .breathing-card { background: white; border-radius: 20px; padding: 30px; text-align: center; max-width: 380px; width: 90%; }
    .circle-container { width: 180px; height: 180px; margin: 20px auto; position: relative; display: flex; align-items: center; justify-content: center; }
    .circle-pacer { width: 140px; height: 140px; border-radius: 50%; background: #C8E6C9; transition: transform 4s ease-in-out; }
    .circle-pacer.inhale { transform: scale(1.3); background: #81C784; transition-duration: 4s; }
    .circle-pacer.hold { transform: scale(1.3); background: #4CAF50; }
    .circle-pacer.exhale { transform: scale(0.7); background: #C8E6C9; transition-duration: 8s; }
    .pacer-label { font-size: 20px; font-weight: 700; color: var(--primary); position: absolute; }
    .btn-interrupt { background: #E0E0E0; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 15px; }
    .btn-interrupt:hover { background: #D5D5D5; }

    /* JIT Identity Form Modal */
    .modal-box { background: white; border-radius: 14px; padding: 24px; width: 90%; max-width: 420px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); }
    .modal-title { font-size: 16px; font-weight: 700; color: var(--primary); margin-bottom: 6px; }
    .form-group { margin-bottom: 12px; }
    .form-group label { display: block; font-size: 12px; font-weight: 600; color: #444; margin-bottom: 4px; }
    .form-group input, .form-group select { width: 100%; padding: 10px 12px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; }
    .btn-bind { width: 100%; background: var(--primary); color: white; border: none; padding: 12px; border-radius: 6px; font-weight: 700; cursor: pointer; margin-top: 6px; }

    /* Emergency Override Modal */
    .emergency-box { background: var(--danger-bg); border: 2px solid var(--danger); border-radius: 16px; padding: 26px; text-align: center; max-width: 440px; width: 90%; }
    .emergency-title { color: var(--danger); font-size: 20px; font-weight: 800; margin-bottom: 10px; }
    .emergency-text { font-size: 13px; line-height: 1.5; color: #444; margin-bottom: 20px; }
    .contact-btn { display: block; background: var(--danger); color: white; padding: 12px; border-radius: 8px; font-weight: 700; text-decoration: none; margin-bottom: 8px; }

    @keyframes pulse {
      0% { opacity: 1; }
      50% { opacity: 0.6; }
      100% { opacity: 1; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-title">
      <h2 style="font-size: 18px;" id="headerAppTitle">دائرة الصحة أبوظبي - المساعد الصوتي السريري</h2>
      <span class="header-badge">DoH JAWDA</span>
    </div>
    
    <div class="header-actions">
      <!-- Language Switcher -->
      <div class="lang-switcher">
        <button class="lang-btn active" id="btnLangAr" onclick="setLanguage('ar-AE')">🇦🇪 العربية</button>
        <button class="lang-btn" id="btnLangEn" onclick="setLanguage('en-US')">🇬🇧 English</button>
      </div>

      <!-- Link to Clinician Reasoning Inspector -->
      <a href="/observer" class="btn-inspector" id="linkInspector" target="_blank">
        <span>🔍 فاحص التفكير السريري المباشر</span>
      </a>

      <div style="font-size: 11px; opacity: 0.9; font-family: monospace;">
        ID: <span id="lblSessionId" style="font-weight: 700;">loading...</span>
      </div>
    </div>
  </header>

  <div class="layout">
    <!-- Chat Section -->
    <div class="chat-container">
      <div class="messages-area" id="messagesArea">
        <div class="message agent" id="initialGreetingMsg">
          مرحباً بك. أنا المساعد السريري المعتمد من دائرة الصحة بأبوظبي. كيف تشعر اليوم وما هي الأعراض التي تود مشاركتها؟
          <div class="prosody">Voice: ar-AE-FatimaNeural | Style: calm</div>
        </div>
      </div>

      <!-- Controls -->
      <div class="controls-footer">
        <div class="biomarker-bar">
          <div class="slider-group">
            <span id="lblArousalTitle">Arousal (الاستثارة):</span>
            <input type="range" id="sliderArousal" min="0" max="1" step="0.05" value="0.5">
            <span id="valArousal" style="width: 32px; font-weight: 700;">0.50</span>
          </div>
          <div class="slider-group">
            <span id="lblValenceTitle">Valence (المزاج):</span>
            <input type="range" id="sliderValence" min="0" max="1" step="0.05" value="0.5">
            <span id="valValence" style="width: 32px; font-weight: 700;">0.50</span>
          </div>
          <div class="presets">
            <button class="btn-preset" onclick="setAcoustics(0.5, 0.5)">طبيعي</button>
            <button class="btn-preset" onclick="setAcoustics(0.85, 0.25)">قلق مرتفع</button>
            <button class="btn-preset" onclick="setAcoustics(0.25, 0.25)">اكتئاب</button>
          </div>
        </div>

        <div class="input-row">
          <input type="text" id="txtInput" placeholder="اكتب حالتك الصحية أو ما تشعر به..." onkeydown="if(event.key==='Enter') sendTurn()">
          <button class="btn-send" id="btnSend" onclick="sendTurn()">إرسال</button>
        </div>
      </div>

      <!-- 4-7-8 Breathing Overlay -->
      <div class="overlay" id="overlayBreathing">
        <div class="breathing-card">
          <h3 style="color: var(--primary);" id="titleBreathing">Interactive Grounding Mode (4-7-8)</h3>
          <p style="font-size: 13px; color: #666; margin-top: 4px;" id="descBreathing">Follow the circle to regulate autonomic arousal</p>
          <div class="circle-container">
            <div class="circle-pacer" id="pacerCircle"></div>
            <div class="pacer-label" id="pacerText">Inhale (4s)</div>
          </div>
          <button class="btn-interrupt" id="btnInterrupt" onclick="interruptBreathing()">Return to Chat</button>
        </div>
      </div>

      <!-- JIT Identity Form Modal -->
      <div class="overlay" id="overlayIdentity">
        <div class="modal-box">
          <div class="modal-title" id="titleIdentityModal">Just-In-Time Identity Verification (Gate 4)</div>
          <p style="font-size: 12px; color: #666; margin-bottom: 14px;" id="descIdentityModal">Emirates ID is securely bound into decoupled zero-knowledge storage to hold your appointment.</p>
          <div class="form-group">
            <label id="lblFullName">الاسم الكامل (Legal Full Name)</label>
            <input type="text" id="idFullName" value="Sultan Al Dhaheri">
          </div>
          <div class="form-group">
            <label id="lblEid">رقم الهوية الإماراتية (784-YYYY-XXXXXXX-Z)</label>
            <input type="text" id="idEmiratesId" value="784-1990-1234567-1">
          </div>
          <div class="form-group">
            <label id="lblCarrier">شركة التأمين (Insurance Carrier)</label>
            <select id="idInsuranceCarrier">
              <option value="DAMAN_ENHANCED" selected>DAMAN Enhanced</option>
              <option value="DAMAN_BASIC">DAMAN Basic</option>
              <option value="THIQA">Thiqa</option>
            </select>
          </div>
          <div class="form-group">
            <label id="lblMemberId">رقم عضوية التأمين (Member ID)</label>
            <input type="text" id="idMemberId" value="DAMAN-99210-AE">
          </div>
          <button class="btn-bind" id="btnConfirmId" onclick="submitIdentity()">Confirm & Bind Identity</button>
        </div>
      </div>

      <!-- Emergency Red Lockout Modal -->
      <div class="overlay" id="overlayEmergency">
        <div class="emergency-box">
          <div class="emergency-title">⚠️ HARD EMERGENCY OVERRIDE</div>
          <div class="emergency-text" id="emergencyDesc">
            سلامتك هي أولويتنا المطلقة الآن. تم إيقاف المحادثة لمساعدتك على التواصل فوراً مع خطوط الدعم المتخصصة في أبوظبي.
          </div>
          <a class="contact-btn" href="tel:800968">Call SAKINA Hotline (800-968)</a>
          <a class="contact-btn" style="background: #333;" href="tel:8001717">Call Estijaba Crisis Services (800-1717)</a>
        </div>
      </div>
    </div>

    <!-- Telemetry Sidebar -->
    <div class="telemetry-sidebar">
      <div class="card">
        <div class="card-title">
          <span id="titleTriageStatus">الحالة السريرية (Triage Status)</span>
          <span class="telemetry-badge badge-tier-1" id="badgeTier">TIER_1_SELF_CARE</span>
        </div>
        <div class="score-meter">
          <div class="score-box">
            <div class="score-num" id="lblPhq9">0</div>
            <div class="score-lbl">PHQ-9 الاكتئاب</div>
          </div>
          <div class="score-box">
            <div class="score-num" id="lblGad7">0</div>
            <div class="score-lbl">GAD-7 القلق</div>
          </div>
          <div class="score-box">
            <div class="score-num" id="lblCssrs" style="color: #2E7D32;">Safe</div>
            <div class="score-lbl">C-SSRS السلامة</div>
          </div>
        </div>
        <div style="font-size: 12px; color: #555; display: flex; justify-content: space-between;">
          <span>الوضع الحالي: <b id="lblMode">CONVERSATIONAL</b></span>
          <span>الأعراض الجسدية: <b id="lblSomaticCount">0</b></span>
        </div>
      </div>

      <!-- Consent & Report Simulation -->
      <div class="card">
        <div class="card-title">مصادقة الطبيب والموافقة (Gate 5)</div>
        <div style="font-size: 12px; color: #666; margin-bottom: 8px;">رمز الوصول (Token): <b id="lblAccessToken" style="font-family: monospace; color: var(--primary);">NOT GENERATED</b></div>
        <div style="display: flex; gap: 6px;">
          <input type="text" id="txtDoctorToken" placeholder="Doctor inputs DOH-XXXX" style="flex: 1; padding: 8px; border: 1px solid var(--border); border-radius: 6px; font-size: 12px;">
          <button onclick="doctorRequestAccess()" style="background: var(--primary); color: white; border: none; padding: 0 12px; border-radius: 6px; font-size: 12px; cursor: pointer;">طلب الملف</button>
        </div>
        <div id="consentChallengeBox" style="display: none; background: #FFF9C4; border: 1px solid #FBC02D; border-radius: 8px; padding: 10px; margin-top: 10px;">
          <div style="font-size: 12px; font-weight: 700; color: #7F0000;" id="consentChallengeText">Dr. Maryam requests access to your clinical DiagnosticReport.</div>
          <div style="display: flex; gap: 8px; margin-top: 8px;">
            <button onclick="patientAuthorize(true)" style="background: #2E7D32; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-size: 11px; cursor: pointer;">موافقة (Approve)</button>
            <button onclick="patientAuthorize(false)" style="background: #C62828; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-size: 11px; cursor: pointer;">رفض (Deny)</button>
          </div>
        </div>
        <pre id="fhirPreview" style="display: none; background: #263238; color: #80CBC4; padding: 8px; border-radius: 6px; font-size: 10px; max-height: 140px; overflow: auto; margin-top: 8px;"></pre>
      </div>

      <!-- Longitudinal Milestones -->
      <div class="card">
        <div class="card-title">مؤشرات جودة الرعاية (Gate 6)</div>
        <ul style="font-size: 12px; color: #444; list-style: none; display: flex; flex-direction: column; gap: 8px;">
          <li style="display: flex; gap: 8px; align-items: center;">
            <span>📅</span> <span><b>اليوم -1:</b> دليل الإعداد للزيارة وتخفيف القلق</span>
          </li>
          <li style="display: flex; gap: 8px; align-items: center;">
            <span>⏰</span> <span><b>قبل ساعة:</b> تسجيل سريع ورابط غرفة العيادة</span>
          </li>
          <li style="display: flex; gap: 8px; align-items: center;">
            <span>📊</span> <span><b>اليوم +30:</b> تدقيق مؤشر جودة (Jawda KPI Δ ≥ 50%)</span>
          </li>
        </ul>
      </div>
    </div>
  </div>

  <script>
    const sessionId = "anon_ghost_" + Math.random().toString(36).substring(2, 10);
    document.getElementById("lblSessionId").innerText = sessionId;
    document.getElementById("linkInspector").href = `/observer/${sessionId}`;

    let currentLang = "ar-AE";
    let ws;
    let breathingInterval;

    function setLanguage(lang) {
      currentLang = lang;
      const html = document.getElementById("htmlRoot");
      const btnAr = document.getElementById("btnLangAr");
      const btnEn = document.getElementById("btnLangEn");

      if (lang === "ar-AE") {
        html.setAttribute("lang", "ar");
        html.setAttribute("dir", "ltr");
        btnAr.classList.add("active");
        btnEn.classList.remove("active");
        
        document.getElementById("headerAppTitle").innerText = "دائرة الصحة أبوظبي - المساعد الصوتي السريري";
        document.getElementById("linkInspector").innerHTML = "<span>🔍 فاحص التفكير السريري المباشر</span>";
        if (document.getElementById("txtInput").disabled) {
          document.getElementById("txtInput").placeholder = "🔒 تم اعتماد مسار الرعاية - المحادثة مغلقة.";
        } else {
          document.getElementById("txtInput").placeholder = "اكتب حالتك الصحية أو ما تشعر به...";
        }
        document.getElementById("btnSend").innerText = "إرسال";
        document.getElementById("lblArousalTitle").innerText = "Arousal (الاستثارة):";
        document.getElementById("lblValenceTitle").innerText = "Valence (المزاج):";
        document.getElementById("titleTriageStatus").innerText = "الحالة السريرية (Triage Status)";
      } else {
        html.setAttribute("lang", "en");
        html.setAttribute("dir", "ltr");
        btnEn.classList.add("active");
        btnAr.classList.remove("active");

        document.getElementById("headerAppTitle").innerText = "Abu Dhabi Health Authority - Clinical Triage";
        document.getElementById("linkInspector").innerHTML = "<span>🔍 Open Live Reasoning Inspector</span>";
        if (document.getElementById("txtInput").disabled) {
          document.getElementById("txtInput").placeholder = "🔒 Clinical triage finalized - Chat session closed.";
        } else {
          document.getElementById("txtInput").placeholder = "Describe how you are feeling or what symptoms you have...";
        }
        document.getElementById("btnSend").innerText = "Send";
        document.getElementById("lblArousalTitle").innerText = "Arousal:";
        document.getElementById("lblValenceTitle").innerText = "Valence:";
        document.getElementById("titleTriageStatus").innerText = "Triage Status";
      }
    }

    function initWebSocket() {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(`${protocol}//${window.location.host}/ws/triage/${sessionId}`);
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
      };

      ws.onclose = () => {
        setTimeout(initWebSocket, 2000);
      };
    }

    function handleServerMessage(data) {
      if (data.type === "AGENT_TURN") {
        appendMessage("agent", data.reply_text, data.ssml_tags);
        if (data.dashboard_telemetry) {
          updateDashboard(data.dashboard_telemetry);
        }
        if (data.dispatched_action) {
          handleAction(data.dispatched_action);
        }
      } else if (data.type === "SESSION_FINALIZED") {
        freezeChatSession(data.triage_tier || "FINALIZED");
        appendMessage("agent", data.reply_text);
      } else if (data.type === "IDENTITY_BOUND_SUCCESS") {
        document.getElementById("overlayIdentity").classList.remove("active");
        document.getElementById("lblAccessToken").innerText = data.access_token;
        document.getElementById("txtDoctorToken").value = data.access_token;
        appendMessage("agent", `✅ تم توثيق الهوية وحجز الموعد بنجاح. رمز الوصول المعتمد: ${data.access_token}`);
      } else if (data.type === "CONSENT_REQUEST_CHALLENGE") {
        document.getElementById("consentChallengeBox").style.display = "block";
        document.getElementById("consentChallengeText").innerText = `${data.doctor} at ${data.facility} requests access to your clinical report.`;
      } else if (data.type === "ERROR") {
        alert(data.message);
      }
    }

    function freezeChatSession(tier) {
      const txt = document.getElementById("txtInput");
      const btn = document.getElementById("btnSend");
      if (txt) {
        txt.disabled = true;
        txt.placeholder = (currentLang === "ar-AE")
          ? "🔒 تم اعتماد مسار الرعاية (" + tier + ") - المحادثة مغلقة."
          : "🔒 Clinical triage finalized (" + tier + ") - Chat session closed.";
        txt.style.background = "#F1F5F9";
        txt.style.cursor = "not-allowed";
      }
      if (btn) {
        btn.disabled = true;
        btn.style.opacity = "0.5";
        btn.style.cursor = "not-allowed";
      }

      const area = document.getElementById("messagesArea");
      const banner = document.createElement("div");
      banner.className = "message-status-banner";
      banner.style.cssText = "align-self: center; background: #FEF3C7; border: 1px solid #F59E0B; color: #92400E; padding: 10px 18px; border-radius: 8px; font-size: 13px; font-weight: 700; text-align: center; margin: 12px 0; width: 90%;";
      banner.innerText = (currentLang === "ar-AE")
        ? `🏁 اكتمل التقييم السريري: تم اعتماد مسار ${tier}. تم إيقاف المحادثة لضمان تنفيذ مسار الرعاية.`
        : `🏁 Clinical assessment completed: ${tier} locked. Chat stopped to execute the care pathway.`;
      area.appendChild(banner);
      area.scrollTop = area.scrollHeight;
    }

    function unfreezeChatSession() {
      const txt = document.getElementById("txtInput");
      const btn = document.getElementById("btnSend");
      if (txt) {
        txt.disabled = false;
        txt.placeholder = (currentLang === "ar-AE")
          ? "اكتب حالتك الصحية أو ما تشعر به..."
          : "Describe how you are feeling or what symptoms you have...";
        txt.style.background = "white";
        txt.style.cursor = "text";
      }
      if (btn) {
        btn.disabled = false;
        btn.style.opacity = "1";
        btn.style.cursor = "pointer";
      }
    }

    function handleAction(action) {
      if (action.action_type === "SWITCH_UI_MODE" && action.payload.target_mode === "BREATHING_EXERCISE") {
        freezeChatSession("LEVEL 1 (BREATHING_EXERCISE)");
        startBreathing();
      } else if (action.action_type === "TRIGGER_JIT_IDENTITY_GATE") {
        freezeChatSession("LEVEL 3 (IN_CLINIC_VISIT)");
        document.getElementById("overlayIdentity").classList.add("active");
        document.getElementById("titleIdentityModal").innerText = "Just-In-Time Identity Verification (Gate 4)";
        document.getElementById("descIdentityModal").innerText = "Emirates ID is securely bound into decoupled zero-knowledge storage to hold your appointment.";
      } else if (action.action_type === "INITIATE_TELEHEALTH_HANDOFF") {
        freezeChatSession("LEVEL 2 (TELEHEALTH_VIDEO)");
        document.getElementById("overlayIdentity").classList.add("active");
        document.getElementById("titleIdentityModal").innerText = "SEHA Telehealth Virtual Care (Gate 4)";
        document.getElementById("descIdentityModal").innerText = "Level 2 Mild-to-Moderate depression/anxiety indicated. Connect your Emirates ID to unlock your SEHA video room link: " + (action.payload.room_link || "https://telehealth.seha.ae/room/virtual-care");
      } else if (action.action_type === "HARD_EMERGENCY_OVERRIDE") {
        freezeChatSession("EMERGENCY (TIER_EMERGENCY)");
        document.getElementById("overlayEmergency").classList.add("active");
        if (action.payload.dialogue) {
          document.getElementById("emergencyDesc").innerText = action.payload.dialogue;
        }
        const badge = document.getElementById("badgeTier");
        badge.className = "telemetry-badge badge-emergency";
        badge.innerText = "TIER_EMERGENCY";
      } else if (action.action_type === "RESUME_CONVERSATION") {
        unfreezeChatSession();
      }
    }

    function sendTurn() {
      const txt = document.getElementById("txtInput");
      if (!txt || txt.disabled) return;
      const transcript = txt.value.trim();
      if (!transcript || !ws || ws.readyState !== WebSocket.OPEN) return;

      appendMessage("user", transcript);
      txt.value = "";

      const arousal = parseFloat(document.getElementById("sliderArousal").value);
      const valence = parseFloat(document.getElementById("sliderValence").value);

      ws.send(JSON.stringify({
        transcript: transcript,
        arousal: arousal,
        valence: valence,
        language: currentLang
      }));
    }

    function appendMessage(role, text, ssml) {
      const area = document.getElementById("messagesArea");
      const msg = document.createElement("div");
      msg.className = `message ${role}`;
      msg.innerText = text;

      if (ssml) {
        const p = document.createElement("div");
        p.className = "prosody";
        p.innerText = `Voice: ${ssml.voice} | Style: ${ssml.style} (${ssml.rate}, ${ssml.pitch})`;
        msg.appendChild(p);
      }

      area.appendChild(msg);
      area.scrollTop = area.scrollHeight;
    }

    function updateDashboard(t) {
      document.getElementById("lblPhq9").innerText = t.phq9_total;
      document.getElementById("lblGad7").innerText = t.gad7_total;
      document.getElementById("lblSomaticCount").innerText = t.somatic_count;
      document.getElementById("lblMode").innerText = t.active_mode;

      const cssrsElem = document.getElementById("lblCssrs");
      if (cssrsElem) {
        if (t.cssrs_critical) {
          cssrsElem.innerText = "CRITICAL";
          cssrsElem.style.color = "#C62828";
        } else if (t.cssrs_screened || (t.cssrs_count && t.cssrs_count > 0)) {
          cssrsElem.innerText = "Screened (" + t.cssrs_count + ")";
          cssrsElem.style.color = "#E65100";
        } else {
          cssrsElem.innerText = "Safe";
          cssrsElem.style.color = "#2E7D32";
        }
      }

      const badge = document.getElementById("badgeTier");
      badge.innerText = t.triage_tier;
      if (t.triage_tier === "TIER_1_SELF_CARE") badge.className = "telemetry-badge badge-tier-1";
      else if (t.triage_tier === "TIER_2_MILD_MOD") badge.className = "telemetry-badge badge-tier-2";
      else if (t.triage_tier === "TIER_3_SEVERE") badge.className = "telemetry-badge badge-tier-3";
      else if (t.triage_tier === "TIER_EMERGENCY") badge.className = "telemetry-badge badge-emergency";
    }

    function setAcoustics(a, v) {
      document.getElementById("sliderArousal").value = a;
      document.getElementById("sliderValence").value = v;
      document.getElementById("valArousal").innerText = a.toFixed(2);
      document.getElementById("valValence").innerText = v.toFixed(2);
    }

    document.getElementById("sliderArousal").oninput = (e) => {
      document.getElementById("valArousal").innerText = parseFloat(e.target.value).toFixed(2);
    };
    document.getElementById("sliderValence").oninput = (e) => {
      document.getElementById("valValence").innerText = parseFloat(e.target.value).toFixed(2);
    };

    // 4-7-8 Breathing Loop
    function startBreathing() {
      document.getElementById("overlayBreathing").classList.add("active");
      const circle = document.getElementById("pacerCircle");
      const lbl = document.getElementById("pacerText");

      function cycle() {
        lbl.innerText = "Inhale (4s)";
        circle.className = "circle-pacer inhale";
        setTimeout(() => {
          lbl.innerText = "Hold (7s)";
          circle.className = "circle-pacer hold";
          setTimeout(() => {
            lbl.innerText = "Exhale (8s)";
            circle.className = "circle-pacer exhale";
          }, 7000);
        }, 4000);
      }
      cycle();
      breathingInterval = setInterval(cycle, 19000);
    }

    function interruptBreathing() {
      clearInterval(breathingInterval);
      document.getElementById("overlayBreathing").classList.remove("active");
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "USER_INTERRUPT_MODE" }));
      }
    }

    async function submitIdentity() {
      const fullName = document.getElementById("idFullName").value;
      const eid = document.getElementById("idEmiratesId").value;
      const carrier = document.getElementById("idInsuranceCarrier").value;
      const memberId = document.getElementById("idMemberId").value;

      try {
        const res = await fetch(`/api/session/${sessionId}/bind-identity`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            full_name: fullName,
            emirates_id: eid,
            insurance_carrier: carrier,
            insurance_member_id: memberId
          })
        });
        const data = await res.json();
        if (!res.ok) alert(data.detail || "Validation Error");
      } catch (err) {
        alert("Error binding identity: " + err);
      }
    }

    async function doctorRequestAccess() {
      const token = document.getElementById("txtDoctorToken").value.trim();
      if (!token) return alert("Enter token");
      try {
        const res = await fetch(`/api/reports/request-access/${token}`);
        const data = await res.json();
        if (res.ok) {
          alert("Challenge dispatched to patient screen!");
        } else {
          alert(data.detail);
        }
      } catch (err) {
        alert(err);
      }
    }

    async function patientAuthorize(granted) {
      const token = document.getElementById("txtDoctorToken").value.trim() || document.getElementById("lblAccessToken").innerText;
      try {
        const res = await fetch(`/api/reports/authorize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ access_token: token, granted: granted })
        });
        const data = await res.json();
        document.getElementById("consentChallengeBox").style.display = "none";
        if (granted && data.fhir_diagnostic_report) {
          const pre = document.getElementById("fhirPreview");
          pre.style.display = "block";
          pre.innerText = JSON.stringify(data.fhir_diagnostic_report, null, 2);
        }
      } catch (err) {
        alert(err);
      }
    }

    window.onload = () => {
      initWebSocket();
      setLanguage("ar-AE");
    };
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
