"""
state.py - TypedDict state schemas for the Clinical Agent Engine (Person 2).
Adheres to Abu Dhabi DoH Jawda-Compliant Clinical Voice Navigation specifications.
"""

from typing import TypedDict, Optional, List, Dict, Any, Literal


class ClinicalSlot(TypedDict):
    score: int  # 0 to 3
    evidence: str


class SomaticSymptom(TypedDict):
    symptom: str
    snomed_ct: str
    rule_out_flag: bool


class SSMLTags(TypedDict):
    voice: str
    style: str
    styledegree: str
    rate: str
    pitch: str
    pause_duration: str


class DispatchedAction(TypedDict):
    action_type: str
    payload: Dict[str, Any]


class ClinicalGraphState(TypedDict):
    session_id: str
    language: str  # "ar-AE" | "en-US"
    active_mode: str  # "CONVERSATIONAL" | "BREATHING_EXERCISE" | "EMERGENCY_LOCK"
    messages: List[Dict[str, str]]
    vocal_arousal: float  # 0.0 to 1.0
    vocal_valence: float  # 0.0 to 1.0
    affect_discrepancy: bool
    phq9_slots: Dict[str, ClinicalSlot]
    phq9_total: int  # 0 to 27
    gad7_slots: Dict[str, ClinicalSlot]
    gad7_total: int  # 0 to 21
    cssrs_slots: Dict[str, ClinicalSlot]  # q1_wish_dead, q2_suicidal_thoughts, q3_suicidal_methods, q4_suicidal_intent, q5_suicidal_plan, q6_suicidal_behavior
    cssrs_critical: bool
    cssrs_screened: bool
    somatic_flags: List[SomaticSymptom]
    triage_tier: str  # "TIER_EMERGENCY" | "TIER_1_SELF_CARE" | "TIER_2_MILD_MOD" | "TIER_3_SEVERE"
    care_modality: str  # "EMERGENCY_HOTLINE" | "EXERCISE" | "TELEHEALTH_VIDEO" | "IN_CLINIC_VISIT" | "CONVERSATIONAL"
    reply_text: str
    ssml_tags: SSMLTags
    dispatched_action: Optional[DispatchedAction]
    scheduled_reminders: List[Dict[str, Any]]
    agentic_reasoning: Optional[Dict[str, Any]]
