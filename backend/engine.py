"""
engine.py - LangGraph StateGraph, nodes, and conditional edges for Person 2.
Implements C-SSRS safety circuit breaker, official PHQ-9/GAD-7 extraction,
stepped-care classification, JIT identity gating, milestone scheduling, and SSML prosody synthesis.
Optimized for low-latency (< 1s) single-pass clinical cognition.
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

load_dotenv()

from backend.state import ClinicalGraphState, ClinicalSlot, SomaticSymptom, SSMLTags, DispatchedAction
from backend.prompts import (
    UNIFIED_CLINICAL_COGNITION_PROMPT,
    SAFETY_EVAL_PROMPT,
    EXTRACTION_PROMPT,
    DIALOGUE_PROMPT
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PRIMARY_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
FALLBACK_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "allam-2-7b", "openai/gpt-oss-20b"]


def get_llm(temperature: float = 0.1, json_mode: bool = False):
    """Instantiate ChatGroq with robust fallback model support."""
    kwargs = {
        "api_key": GROQ_API_KEY,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    models_to_try = [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m != PRIMARY_MODEL]
    for model_name in models_to_try:
        try:
            return ChatGroq(model=model_name, **kwargs)
        except Exception:
            continue
    return ChatGroq(model="openai/gpt-oss-120b", **kwargs)


def _safe_parse_json(text: str) -> dict:
    """Robustly extract and parse JSON from LLM output."""
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except Exception:
                pass
    return {}


# -------------------------------------------------------------------------
# NODES
# -------------------------------------------------------------------------

def evaluate_safety(state: ClinicalGraphState) -> Dict[str, Any]:
    """
    Gate 0: Semantic C-SSRS Safety Evaluation.
    Differentiates medication compliance/factual statements from suicidal ideation/overdose intent.
    Fast deterministic regex check provides instant evaluation.
    """
    messages = state.get("messages", [])
    user_utterance = messages[-1].get("content", "") if messages else ""

    # Fast-check for medication adherence
    adherence_patterns = [
        r"take (?:all )?(?:my )?(?:prescribed )?(?:anxiety |blood pressure |daily )?pills (?:every |on |with )",
        r"take (?:my )?medicine on time",
        r"taking my medication as prescribed",
        r"take all my anxiety pills on time"
    ]
    is_explicit_adherence = any(re.search(pat, user_utterance.lower()) for pat in adherence_patterns)

    # Fast-check for critical suicide intent
    suicide_patterns = [
        r"took (?:all )?(?:my )?pills to end (?:my life|it)",
        r"swallowed all (?:my )?pills to end",
        r"end my life",
        r"kill myself",
        r"want to die",
        r"don't want to wake up",
        r"sleep forever",
        r"أنهي حياتي",
        r"أريد الموت",
        r"بلعت الحبوب لأموت"
    ]
    is_explicit_suicide = any(re.search(pat, user_utterance.lower()) for pat in suicide_patterns)

    # Check PHQ-9 Item 9 score
    phq9 = state.get("phq9_slots", {})
    item9 = phq9.get("item9_suicide", {}).get("score", 0)

    # Check C-SSRS slots & critical state
    cssrs = state.get("cssrs_slots", {})
    has_cssrs_intent = any(k in cssrs for k in ["q4_intent", "q5_plan", "q6_behavior"])

    is_critical = False
    rationale = ""
    if is_explicit_suicide or item9 >= 2 or has_cssrs_intent or state.get("cssrs_critical"):
        is_critical = True
        rationale = "Critical C-SSRS intent/fatalistic overdose detected."
    elif is_explicit_adherence:
        is_critical = False
        rationale = "Medication adherence / factual compliance recognized. Gate 0 cleared."
    else:
        is_critical = False
        rationale = "Standard conversational turn evaluated as safe."

    reasoning = dict(state.get("agentic_reasoning", {}) or {})
    reasoning["gate_0_safety"] = {
        "status": "CRITICAL_TRIGGERED" if is_critical else "PASSED",
        "cssrs_critical": is_critical,
        "rationale": rationale,
        "evaluated_utterance": user_utterance
    }

    if is_critical:
        return {
            "cssrs_critical": True,
            "triage_tier": "TIER_EMERGENCY",
            "care_modality": "EMERGENCY_HOTLINE",
            "active_mode": "EMERGENCY_LOCK",
            "agentic_reasoning": reasoning
        }
    return {
        "cssrs_critical": False,
        "agentic_reasoning": reasoning
    }


def emergency_cutoff(state: ClinicalGraphState) -> Dict[str, Any]:
    """
    Action 0: Hard Emergency Override.
    Halts audio playback, locks session, and returns emergency hotline payload.
    """
    is_arabic = (state.get("language", "ar-AE") == "ar-AE")
    dialogue = (
        "سلامتك هي أولويتنا المطلقة الآن. سأوقف المحادثة هنا لمساعدتك على التواصل فوراً مع فريق الدعم المختص."
        if is_arabic else
        "Your safety is our absolute priority. I am pausing our conversation now so you can connect immediately with dedicated crisis support."
    )

    action: DispatchedAction = {
        "action_type": "HARD_EMERGENCY_OVERRIDE",
        "payload": {
            "dialogue": dialogue,
            "status": "AUDIO_TERMINATED",
            "ui_lock": True,
            "ui_mode": "EMERGENCY_LOCK",
            "contacts": [
                {"name": "SAKINA Mental Health Hotline", "number": "800968", "tag": "800-SAKINA"},
                {"name": "Estijaba Crisis Services", "number": "8001717", "tag": "DoH 24/7"}
            ]
        }
    }

    ssml: SSMLTags = {
        "voice": "ar-AE-FatimaNeural" if is_arabic else "en-US-JennyMultilingualNeural",
        "style": "calm",
        "styledegree": "1.0",
        "rate": "-12%",
        "pitch": "-4%",
        "pause_duration": "400ms"
    }

    return {
        "reply_text": dialogue,
        "ssml_tags": ssml,
        "dispatched_action": action,
        "active_mode": "EMERGENCY_LOCK",
        "triage_tier": "TIER_EMERGENCY",
        "care_modality": "EMERGENCY_HOTLINE"
    }


def extract_somatic_slots(state: ClinicalGraphState) -> Dict[str, Any]:
    """
    Gate 1: Fast Single-Pass Clinical Cognition & Questionnaire Extraction.
    Uses official PHQ-9 (0-3), GAD-7 (0-3), and C-SSRS rules:
    - Single adjectives like 'demotivated' or 'tired' score at most 1 (several days), NOT 2!
    - Proactively extracts and balances PHQ-9, GAD-7, and C-SSRS slots.
    - Generates empathetic reply with natural clinical steering under 25 words.
    """
    messages = state.get("messages", [])
    user_utterance = messages[-1].get("content", "") if messages else ""
    history_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages[-4:-1]])
    arousal = state.get("vocal_arousal", 0.5)
    valence = state.get("vocal_valence", 0.5)
    lang = state.get("language", "ar-AE")

    phq9 = dict(state.get("phq9_slots", {}))
    gad7 = dict(state.get("gad7_slots", {}))
    cssrs = dict(state.get("cssrs_slots", {}))
    somatic_list = list(state.get("somatic_flags", []))
    affect_discrepancy = state.get("affect_discrepancy", False)

    # Check if pure greeting
    is_pure_greeting = user_utterance.strip().lower() in [
        "hi", "hello", "hey", "good morning", "good evening", "howdy",
        "مرحبا", "مرحباً", "السلام عليكم", "أهلا", "أهلاً", "صباح الخير", "مساء الخير"
    ]

    reply_text = ""
    extraction_rationale = ""

    if is_pure_greeting:
        extraction_rationale = "Initial greeting turn. Instruments held at baseline 0."
        reply_text = (
            "مرحباً بك. أنا المساعد السريري المعتمد من دائرة الصحة بأبوظبي. كيف تشعر اليوم وما الذي يقلقك؟"
            if lang == "ar-AE" else
            "Hello! Welcome to Abu Dhabi Clinical Triage. How are you feeling today, and what brings you in?"
        )
    else:
        try:
            llm = get_llm(temperature=0.2, json_mode=True)
            cssrs_summary = "Screened: " + str(bool(cssrs))
            if cssrs:
                cssrs_summary += " | Items: " + ", ".join(cssrs.keys())

            prompt = UNIFIED_CLINICAL_COGNITION_PROMPT.format(
                language=lang,
                arousal=arousal,
                valence=valence,
                phq9_known=json.dumps(phq9),
                gad7_known=json.dumps(gad7),
                cssrs_known=cssrs_summary,
                history=history_str if history_str else "None",
                user_utterance=user_utterance
            )
            res = llm.invoke(prompt)
            data = _safe_parse_json(res.content)

            # Check safety update from unified call
            safety_data = data.get("safety", {})
            if safety_data.get("is_critical"):
                state["cssrs_critical"] = True

            # Merge PHQ-9 (monotonic max score accumulation)
            for item, slot in data.get("phq9_updates", {}).items():
                if isinstance(slot, dict):
                    new_score = int(slot.get("score", 0))
                    if new_score > 0:
                        if item in phq9:
                            if new_score > phq9[item]["score"]:
                                phq9[item] = {"score": new_score, "evidence": slot.get("evidence", user_utterance)}
                        else:
                            phq9[item] = {"score": new_score, "evidence": slot.get("evidence", user_utterance)}

            # Merge GAD-7 (monotonic max score accumulation)
            for item, slot in data.get("gad7_updates", {}).items():
                if isinstance(slot, dict):
                    new_score = int(slot.get("score", 0))
                    if new_score > 0:
                        if item in gad7:
                            if new_score > gad7[item]["score"]:
                                gad7[item] = {"score": new_score, "evidence": slot.get("evidence", user_utterance)}
                        else:
                            gad7[item] = {"score": new_score, "evidence": slot.get("evidence", user_utterance)}

            # Merge C-SSRS updates
            cssrs_data = data.get("cssrs_updates", {})
            if isinstance(cssrs_data, dict):
                evidence = cssrs_data.get("evidence", user_utterance)
                if cssrs_data.get("screened"):
                    state["cssrs_screened"] = True
                if cssrs_data.get("q1_wish_dead"):
                    cssrs["q1_wish_dead"] = {"score": 1, "evidence": evidence}
                    state["cssrs_screened"] = True
                if cssrs_data.get("q2_active_thoughts"):
                    cssrs["q2_active_thoughts"] = {"score": 2, "evidence": evidence}
                    state["cssrs_screened"] = True
                if cssrs_data.get("q4_intent") or cssrs_data.get("is_critical"):
                    cssrs["q4_intent"] = {"score": 4, "evidence": evidence}
                    state["cssrs_critical"] = True
                    state["cssrs_screened"] = True

            # Merge somatic updates (STRICTLY ONLY if verified in patient's utterance to prevent hallucinations)
            somatic_grounding_map = {
                "271795006": ["chest tightness", "tightness in chest", "tight chest", "chest pain", "pain in chest", "pressure in chest", "heaviness in chest", "ضيق في صدري", "ضيق بالصدر", "ألم في الصدر", "ثقل في صدري", "صدري يوجعني"],
                "80313002": ["palpitation", "palpitations", "racing heart", "heart racing", "heart pounding", "heart flutter", "rapid heartbeat", "تسارع في نبضات القلب", "خفقان", "دقات قلبي سريعة", "قلبي يدق بسرعة"],
                "282291009": ["shortness of breath", "dyspnea", "hard to breathe", "cannot breathe", "can't breathe", "struggling to breathe", "صعوبة في التنفس", "ضيق تنفس", "ما أقدر أتنفس"],
                "398057008": ["headache", "head ache", "tension headache", "صداع", "ألم في رأسي", "راسي يوجعني"]
            }
            text_lower = user_utterance.lower()
            existing_snomed = {s["snomed_ct"] for s in somatic_list}
            for s in data.get("somatic_updates", []):
                if isinstance(s, dict):
                    snomed = str(s.get("snomed_ct", ""))
                    if snomed in somatic_grounding_map:
                        keywords = somatic_grounding_map[snomed]
                        if any(k in text_lower for k in keywords) and snomed not in existing_snomed:
                            somatic_list.append({
                                "symptom": s.get("symptom", "Somatic Symptom"),
                                "snomed_ct": snomed,
                                "rule_out_flag": bool(s.get("rule_out_flag", False))
                            })
                            existing_snomed.add(snomed)

            affect_discrepancy = bool(data.get("affect_discrepancy", False))
            reply_text = data.get("reply_text", "").strip()
            extraction_rationale = f"Evaluated turn. Active PHQ-9 items: {len(phq9)}, GAD-7 items: {len(gad7)}, C-SSRS items: {len(cssrs)}."
        except Exception as e:
            extraction_rationale = f"Heuristic slot evaluation active ({e})."

    # Deterministic Somatic Safety Check (Only triggers if explicitly in text)
    text_lower = user_utterance.lower()
    chest_tightness = any(k in text_lower for k in ["chest tightness", "tightness in chest", "tight chest", "ضيق في صدري", "ضيق بالصدر", "ألم في الصدر"])
    palpitations = any(k in text_lower for k in ["palpitation", "palpitations", "racing heart", "heart racing", "تسارع في نبضات القلب", "خفقان"])
    dyspnea = any(k in text_lower for k in ["shortness of breath", "dyspnea", "صعوبة في التنفس"])

    existing_snomed = {s["snomed_ct"] for s in somatic_list}
    if chest_tightness and "271795006" not in existing_snomed:
        somatic_list.append({"symptom": "Precordial chest tightness", "snomed_ct": "271795006", "rule_out_flag": True})
    if palpitations and "80313002" not in existing_snomed:
        somatic_list.append({"symptom": "Palpitations", "snomed_ct": "80313002", "rule_out_flag": True})
    if dyspnea and "282291009" not in existing_snomed:
        somatic_list.append({"symptom": "Dyspnea on exertion", "snomed_ct": "282291009", "rule_out_flag": True})

    # Acoustic boosting only when not pure greeting
    if not is_pure_greeting:
        if arousal > 0.80 and "item1_nervous_on_edge" not in gad7:
            gad7["item1_nervous_on_edge"] = {"score": 2, "evidence": "High vocal arousal biomarker (>0.80) indicating sympathetic activation"}

    # Recalculate totals
    phq9_total = sum(s["score"] for s in phq9.values())
    gad7_total = sum(s["score"] for s in gad7.values())

    reasoning = dict(state.get("agentic_reasoning", {}) or {})
    reasoning["gate_1_extraction"] = {
        "phq9_total": phq9_total,
        "phq9_slots": phq9,
        "gad7_total": gad7_total,
        "gad7_slots": gad7,
        "cssrs_slots": cssrs,
        "somatic_symptoms": somatic_list,
        "affect_discrepancy": affect_discrepancy,
        "acoustic_biomarkers": {"arousal": arousal, "valence": valence},
        "rationale": extraction_rationale
    }

    return {
        "phq9_slots": phq9,
        "phq9_total": phq9_total,
        "gad7_slots": gad7,
        "gad7_total": gad7_total,
        "cssrs_slots": cssrs,
        "cssrs_screened": bool(state.get("cssrs_screened", False) or bool(cssrs)),
        "somatic_flags": somatic_list,
        "affect_discrepancy": affect_discrepancy,
        "reply_text": reply_text,
        "agentic_reasoning": reasoning
    }


def classify_triage_tier(state: ClinicalGraphState) -> Dict[str, Any]:
    """
    Gate 2: Stepped-Care Severity Classification Truth Table (Strict Official Thresholds):
    1. EMERGENCY (TIER_EMERGENCY):
       - Trigger: C-SSRS active ideation == True OR PHQ-9 Item 9 >= 2 OR explicit suicide regex
       - Action: HARD_EMERGENCY_OVERRIDE
       - Modality: EMERGENCY_HOTLINE
    2. LEVEL 3: SEVERE (TIER_3_SEVERE):
       - Trigger: PHQ-9 >= 15 OR GAD-7 >= 15 OR rule_out_flag == True (chest tightness 271795006, palpitations 80313002)
       - Action: TRIGGER_JIT_IDENTITY_GATE
       - Modality: IN_CLINIC_VISIT
    3. LEVEL 2: MEDIUM / MODERATE (TIER_2_MILD_MOD):
       - Trigger: (PHQ-9 in [5..14] OR GAD-7 in [5..14]) AND rule_out_flag == False AND cssrs_critical == False
       - Action: INITIATE_TELEHEALTH_HANDOFF
       - Modality: TELEHEALTH_VIDEO
    4. LEVEL 1: MILD (TIER_1_SELF_CARE):
       - Trigger: PHQ-9 <= 4 AND GAD-7 <= 4 AND rule_out_flag == False AND cssrs_critical == False
       - Action: SWITCH_UI_MODE (target_mode: BREATHING_EXERCISE) for situational distress, or CONVERSATIONAL_INTAKE for baseline intake.
       - Modality: EXERCISE (or CONVERSATIONAL)
    """
    phq = state.get("phq9_total", 0)
    gad = state.get("gad7_total", 0)
    phq_slots = state.get("phq9_slots", {})
    gad_slots = state.get("gad7_slots", {})
    cssrs_slots = state.get("cssrs_slots", {})
    cssrs_screened = bool(state.get("cssrs_screened", False) or bool(cssrs_slots))
    somatic_list = state.get("somatic_flags", [])
    has_cardiac_red_flag = any(s.get("rule_out_flag", False) for s in somatic_list)
    cssrs_critical = bool(state.get("cssrs_critical", False))

    messages = state.get("messages", [])
    user_utterance = messages[-1].get("content", "").lower() if messages else ""

    explicit_breathing_cues = [
        "need some calm", "need calm", "moment to breathe", "want to breathe", "help me breathe",
        "breathing exercise", "calm down", "calming exercise", "guided breathing",
        "أحتاج أهدأ", "أحتاج أتنفس", "تمارين تنفس", "ساعدني أهدأ"
    ]
    has_breathing_request = any(k in user_utterance for k in explicit_breathing_cues)

    user_requests_booking = any(
        phrase in user_utterance
        for phrase in [
            "book appointment", "book the appointment", "see a doctor", "consultation",
            "what should i do", "what do you recommend", "احجز لي", "احجز الموعد", "أبغى موعد", "أريد موعد"
        ]
    )

    # Coverage rule: Do not jump to Level 2/3 non-emergency conclusions until instruments have coverage:
    # 1. C-SSRS safety screened AND both PHQ-9 & GAD-7 probed, OR
    # 2. Patient explicitly requests booking/next steps, OR
    # 3. Intake has conducted 4+ turns.
    is_coverage_complete = (
        (cssrs_screened and len(phq_slots) >= 1 and len(gad_slots) >= 1) or
        user_requests_booking or
        len(messages) >= 6
    )

    rule_matched = ""
    if cssrs_critical:
        tier = "TIER_EMERGENCY"
        modality = "EMERGENCY_HOTLINE"
        rule_matched = "1. Emergency Circuit Breaker: C-SSRS active ideation / crisis detected -> EMERGENCY_HOTLINE."
    elif has_cardiac_red_flag:
        tier = "TIER_3_SEVERE"
        modality = "IN_CLINIC_VISIT"
        rule_matched = "4. Cardiac Red Flag Circuit Breaker: Chest tightness / racing heart -> IN_CLINIC_VISIT."
    elif has_breathing_request:
        tier = "TIER_1_SELF_CARE"
        modality = "EXERCISE"
        rule_matched = "2. Level 1 Mild: Explicit calming/breathing request -> BREATHING_EXERCISE."
    elif not is_coverage_complete:
        # Multi-instrument screening in progress: stay in continuous dialogue
        tier = "TIER_1_SELF_CARE"
        modality = "CONVERSATIONAL"
        rule_matched = (
            f"Intake In Progress: Screening across instruments (PHQ items: {len(phq_slots)}, "
            f"GAD items: {len(gad_slots)}, C-SSRS screened: {cssrs_screened}) -> CONVERSATIONAL."
        )
    elif phq >= 15 or gad >= 15:
        tier = "TIER_3_SEVERE"
        modality = "IN_CLINIC_VISIT"
        rule_matched = f"4. Level 3 Severe Concluded: PHQ ({phq}) >= 15 OR GAD ({gad}) >= 15 -> IN_CLINIC_VISIT."
    elif (5 <= phq <= 14) or (5 <= gad <= 14):
        tier = "TIER_2_MILD_MOD"
        modality = "TELEHEALTH_VIDEO"
        rule_matched = f"3. Level 2 Medium/Moderate Concluded: PHQ ({phq}) or GAD ({gad}) in [5..14] with full screening -> TELEHEALTH_VIDEO."
    else:
        tier = "TIER_1_SELF_CARE"
        modality = "EXERCISE"
        rule_matched = f"2. Level 1 Mild Concluded: (PHQ {phq}, GAD {gad} <= 4, C-SSRS safe) -> BREATHING_EXERCISE."

    reasoning = dict(state.get("agentic_reasoning", {}) or {})
    reasoning["gate_2_classification"] = {
        "phq9_total": phq,
        "gad7_total": gad,
        "has_cardiac_red_flag": has_cardiac_red_flag,
        "cssrs_critical": cssrs_critical,
        "triage_tier": tier,
        "care_modality": modality,
        "rule_matched": rule_matched
    }

    return {
        "triage_tier": tier,
        "care_modality": modality,
        "agentic_reasoning": reasoning
    }


def route_conversational(state: ClinicalGraphState) -> Dict[str, Any]:
    """Conversational intake node when in baseline intake stage."""
    action: DispatchedAction = {
        "action_type": "CONVERSATIONAL_INTAKE",
        "payload": {"active_mode": "CONVERSATIONAL"}
    }
    reasoning = dict(state.get("agentic_reasoning", {}) or {})
    reasoning["gate_3_routing"] = {
        "action_type": "CONVERSATIONAL_INTAKE",
        "target_mode": "CONVERSATIONAL",
        "rationale": "Patient is in conversational assessment. Active dialogue maintained without forced mode switch."
    }
    return {
        "active_mode": "CONVERSATIONAL",
        "dispatched_action": action,
        "agentic_reasoning": reasoning
    }


def route_guided_exercise(state: ClinicalGraphState) -> Dict[str, Any]:
    """Gate 3A: Interactive Grounding Mode (Level 1)."""
    action: DispatchedAction = {
        "action_type": "SWITCH_UI_MODE",
        "payload": {
            "target_mode": "BREATHING_EXERCISE",
            "cadence": "4-7-8",
            "pacer_config": {"inhale": 4, "hold": 7, "exhale": 8, "cycles": 4},
            "allow_voice_interrupt": True
        }
    }
    reasoning = dict(state.get("agentic_reasoning", {}) or {})
    reasoning["gate_3_routing"] = {
        "action_type": "SWITCH_UI_MODE",
        "target_mode": "BREATHING_EXERCISE",
        "rationale": "Activated 4-7-8 Grounding Canvas. Autonomic nervous system regulation initiated."
    }
    return {
        "active_mode": "BREATHING_EXERCISE",
        "dispatched_action": action,
        "agentic_reasoning": reasoning
    }


def route_telehealth_booking(state: ClinicalGraphState) -> Dict[str, Any]:
    """Gate 3B: Virtual Telehealth Routing (Level 2)."""
    action: DispatchedAction = {
        "action_type": "INITIATE_TELEHEALTH_HANDOFF",
        "payload": {
            "facility": "SEHA Virtual Care Platform",
            "consultation_type": "TELEHEALTH_VIDEO",
            "room_link": "https://telehealth.seha.ae/room/virtual-care",
            "status": "LOCKED_AWAITING_ID",
            "instructions": "Level 2 Mild-to-Moderate depression/anxiety indicated. SEHA Telehealth room link generated."
        }
    }
    reasoning = dict(state.get("agentic_reasoning", {}) or {})
    reasoning["gate_3_routing"] = {
        "action_type": "INITIATE_TELEHEALTH_HANDOFF",
        "target_mode": "TELEHEALTH_VIDEO",
        "rationale": "Level 2 Telehealth room prepared. Telehealth handoff dispatched with SEHA virtual room link."
    }
    return {
        "dispatched_action": action,
        "agentic_reasoning": reasoning
    }


def route_in_clinic_booking(state: ClinicalGraphState) -> Dict[str, Any]:
    """Gate 3C: Physical Outpatient Routing (Level 3 or Cardiac Red Flag)."""
    action: DispatchedAction = {
        "action_type": "TRIGGER_JIT_IDENTITY_GATE",
        "payload": {
            "facility": "Al Bateen Healthcare Center (AHS / SEHA)",
            "specialty": "Family Medicine & Behavioral Health",
            "status": "LOCKED_AWAITING_ID"
        }
    }
    reasoning = dict(state.get("agentic_reasoning", {}) or {})
    reasoning["gate_3_routing"] = {
        "action_type": "TRIGGER_JIT_IDENTITY_GATE",
        "target_mode": "IN_CLINIC_VISIT",
        "rationale": "High-acuity or somatic red-flag detected. Reservation assigned to Al Bateen Healthcare Center pending ID binding."
    }
    return {
        "dispatched_action": action,
        "agentic_reasoning": reasoning
    }


def schedule_reminders(state: ClinicalGraphState) -> Dict[str, Any]:
    """
    Gate 6: Register longitudinal pre-visit and Jawda KPI audit reminders.
    """
    modality = state.get("care_modality", "CONVERSATIONAL")
    reminders = list(state.get("scheduled_reminders", []))

    if modality in ("TELEHEALTH_VIDEO", "IN_CLINIC_VISIT") and not reminders:
        reminders = [
            {
                "touchpoint_type": "PRE_VISIT_24H",
                "offset_hours": -24,
                "description": "Orientation Guide to prevent consultation anxiety and reduce no-shows"
            },
            {
                "touchpoint_type": "PRE_VISIT_1H",
                "offset_hours": -1,
                "description": "Fast-Track Check-In & Clinic GPS / WebRTC Room Link"
            },
            {
                "touchpoint_type": "JAWDA_30D",
                "offset_hours": 720,
                "description": "60-second voice check-in calculating symptom delta against DoH Jawda benchmark"
            }
        ]

    return {
        "scheduled_reminders": reminders
    }


def generate_adaptive_dialogue(state: ClinicalGraphState) -> Dict[str, Any]:
    """
    Synthesizes SSML tags based on arousal and valence, and guarantees warm conversational reply.
    Zero additional LLM call needed since reply_text is generated in the unified pass.
    """
    language = state.get("language", "ar-AE")
    is_arabic = (language == "ar-AE")
    modality = state.get("care_modality", "CONVERSATIONAL")
    arousal = state.get("vocal_arousal", 0.5)
    valence = state.get("vocal_valence", 0.5)

    reply_text = state.get("reply_text", "").strip()

    # Fallback only if reply_text was somehow empty
    if not reply_text:
        if is_arabic:
            if modality == "EXERCISE":
                reply_text = "سلامتك يا خوي. أسمعك وحاس بقد إيش الحمل ثقيل عليك، خذ نفس عميق مع الدائرة الظاهرة أمامك."
            elif modality == "IN_CLINIC_VISIT":
                reply_text = "سلامتك يا خوي. عشان نتأكد من سلامتك وصحتك، جهزت لك موعد في مركز البطين الصحي، يرجى إكمال البيانات."
            elif modality == "TELEHEALTH_VIDEO":
                reply_text = "سلامتك يا خوي. أسمعك وحاس فيك، جهزت لك استشارة افتراضية مع طبيب مختص عبر الشاشة."
            else:
                reply_text = "سلامتك يا خوي. أنا معك وأسمعك بكل اهتمام، كيف صحتك اليوم؟"
        else:
            if modality == "EXERCISE":
                reply_text = "I hear you and understand this feels heavy. Please follow the gentle breathing circle opening on your screen."
            elif modality == "IN_CLINIC_VISIT":
                reply_text = "To ensure your physical safety and well-being, I am holding an appointment at Al Bateen Health Center for you."
            elif modality == "TELEHEALTH_VIDEO":
                reply_text = "I hear you and we are here to support you. Let us connect you to a virtual doctor visit on your screen."
            else:
                reply_text = "I am listening closely and here to support you. Please take your time."

    voice = "ar-AE-FatimaNeural" if is_arabic else "en-US-JennyMultilingualNeural"
    if arousal > 0.65:
        style = "calm"
        styledegree = "1.4"
        rate = "-10%"
        pitch = "-4%"
        pause_duration = "350ms"
    elif valence < 0.40:
        style = "empathetic"
        styledegree = "1.2"
        rate = "-8%"
        pitch = "-2%"
        pause_duration = "300ms"
    else:
        style = "friendly"
        styledegree = "1.0"
        rate = "-5%"
        pitch = "0%"
        pause_duration = "250ms"

    ssml: SSMLTags = {
        "voice": voice,
        "style": style,
        "styledegree": styledegree,
        "rate": rate,
        "pitch": pitch,
        "pause_duration": pause_duration
    }

    return {
        "reply_text": reply_text,
        "ssml_tags": ssml
    }


# -------------------------------------------------------------------------
# CONDITIONAL ROUTING FUNCTIONS
# -------------------------------------------------------------------------

def route_safety_check(state: ClinicalGraphState) -> str:
    """Evaluate whether to abort for emergency crisis or continue to slot extraction."""
    if state.get("cssrs_critical", False):
        return "emergency_cutoff"
    return "extract_somatic_slots"


def route_action_modality(state: ClinicalGraphState) -> str:
    """Route according to classified care modality."""
    modality = state.get("care_modality", "CONVERSATIONAL")
    if modality == "EXERCISE":
        return "route_guided_exercise"
    elif modality == "TELEHEALTH_VIDEO":
        return "route_telehealth_booking"
    elif modality == "IN_CLINIC_VISIT":
        return "route_in_clinic_booking"
    return "route_conversational"


# -------------------------------------------------------------------------
# COMPILE STATEGRAPH
# -------------------------------------------------------------------------

def build_clinical_graph() -> StateGraph:
    workflow = StateGraph(ClinicalGraphState)

    workflow.add_node("evaluate_safety", evaluate_safety)
    workflow.add_node("emergency_cutoff", emergency_cutoff)
    workflow.add_node("extract_somatic_slots", extract_somatic_slots)
    workflow.add_node("classify_triage_tier", classify_triage_tier)
    workflow.add_node("route_conversational", route_conversational)
    workflow.add_node("route_guided_exercise", route_guided_exercise)
    workflow.add_node("route_telehealth_booking", route_telehealth_booking)
    workflow.add_node("route_in_clinic_booking", route_in_clinic_booking)
    workflow.add_node("schedule_reminders", schedule_reminders)
    workflow.add_node("generate_adaptive_dialogue", generate_adaptive_dialogue)

    workflow.set_entry_point("evaluate_safety")

    workflow.add_conditional_edges(
        "evaluate_safety",
        route_safety_check,
        {
            "emergency_cutoff": "emergency_cutoff",
            "extract_somatic_slots": "extract_somatic_slots"
        }
    )

    workflow.add_edge("emergency_cutoff", END)
    workflow.add_edge("extract_somatic_slots", "classify_triage_tier")

    workflow.add_conditional_edges(
        "classify_triage_tier",
        route_action_modality,
        {
            "route_conversational": "route_conversational",
            "route_guided_exercise": "route_guided_exercise",
            "route_telehealth_booking": "route_telehealth_booking",
            "route_in_clinic_booking": "route_in_clinic_booking"
        }
    )

    workflow.add_edge("route_conversational", "schedule_reminders")
    workflow.add_edge("route_guided_exercise", "schedule_reminders")
    workflow.add_edge("route_telehealth_booking", "schedule_reminders")
    workflow.add_edge("route_in_clinic_booking", "schedule_reminders")

    workflow.add_edge("schedule_reminders", "generate_adaptive_dialogue")
    workflow.add_edge("generate_adaptive_dialogue", END)

    return workflow.compile()


clinical_engine = build_clinical_graph()
