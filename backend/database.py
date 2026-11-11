"""
database.py - Decoupled zero-knowledge database layer supporting Supabase PostgreSQL
and SQLite fallback for the Clinical Agent Engine (Person 2).
Enforces zero-knowledge PII separation, AES-256-GCM encryption, and tokenized consent.
"""

import os
import json
import time
import uuid
import base64
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

load_dotenv()

# Database connection: reads from DATABASE_URL with SQLite local fallback
SUPABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///clinical_sessions.db"
)

# AES-256 Key for Zero-Knowledge Vault and Clinical Reports
_vault_key_str = os.getenv("VAULT_SECRET_KEY", "prodrome-default-dev-vault-key-32b")
_RAW_KEY = hashlib.sha256(_vault_key_str.encode("utf-8")).digest()
_CIPHER = AESGCM(_RAW_KEY)


def encrypt_data(plaintext: str) -> str:
    """Encrypt plaintext string using AES-256-GCM."""
    nonce = os.urandom(12)
    ciphertext = _CIPHER.encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ciphertext).decode("utf-8")


def decrypt_data(token_b64: str) -> str:
    """Decrypt AES-256-GCM encrypted base64 payload."""
    raw = base64.b64decode(token_b64.encode("utf-8"))
    nonce = raw[:12]
    ciphertext = raw[12:]
    decrypted = _CIPHER.decrypt(nonce, ciphertext, None)
    return decrypted.decode("utf-8")


def hash_session_id(session_id: str) -> str:
    """Generate SHA-256 hash of session_id for zero-knowledge vault decoupling."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


class DatabaseClient:
    """Handles connection and queries for either Supabase Postgres or SQLite."""

    def __init__(self, db_url: str = SUPABASE_URL):
        self.db_url = db_url
        self.is_postgres = not (db_url.startswith("sqlite") or ":memory:" in db_url)

    def get_connection(self):
        if self.is_postgres:
            import psycopg2
            return psycopg2.connect(self.db_url, sslmode="require")
        else:
            import sqlite3
            sqlite_path = self.db_url.replace("sqlite:///", "").replace("sqlite://", "")
            if not sqlite_path or sqlite_path == ":memory:":
                sqlite_path = ":memory:"
            conn = sqlite3.connect(sqlite_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn

    def execute_write(self, query: str, params: tuple = ()) -> None:
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            if not self.is_postgres:
                query = query.replace("%s", "?")
            cur.execute(query, params)
            conn.commit()
        finally:
            conn.close()

    def execute_read_one(self, query: str, params: tuple = ()) -> Optional[dict]:
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            if not self.is_postgres:
                query = query.replace("%s", "?")
            cur.execute(query, params)
            row = cur.fetchone()
            if row is None:
                return None
            if self.is_postgres:
                colnames = [desc[0] for desc in cur.description]
                return dict(zip(colnames, row))
            else:
                return dict(row)
        finally:
            conn.close()

    def execute_read_all(self, query: str, params: tuple = ()) -> List[dict]:
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            if not self.is_postgres:
                query = query.replace("%s", "?")
            cur.execute(query, params)
            rows = cur.fetchall()
            if not rows:
                return []
            if self.is_postgres:
                colnames = [desc[0] for desc in cur.description]
                return [dict(zip(colnames, r)) for r in rows]
            else:
                return [dict(r) for r in rows]
        finally:
            conn.close()


# Default singleton client
_db = DatabaseClient()


def get_db() -> DatabaseClient:
    return _db


def set_db(db_client: DatabaseClient):
    global _db
    _db = db_client


def init_db(db_client: Optional[DatabaseClient] = None) -> None:
    """Create all 4 decoupled tables."""
    client = db_client or _db

    # Sessions Table (Public Domain - Clinical Data Without Legal Identifiers)
    client.execute_write("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        language TEXT NOT NULL DEFAULT 'ar-AE',
        active_mode TEXT NOT NULL DEFAULT 'CONVERSATIONAL',
        triage_tier TEXT NOT NULL DEFAULT 'TIER_1_SELF_CARE',
        care_modality TEXT NOT NULL DEFAULT 'CONVERSATIONAL',
        phq9_json TEXT NOT NULL DEFAULT '{}',
        gad7_json TEXT NOT NULL DEFAULT '{}',
        cssrs_json TEXT NOT NULL DEFAULT '{}',
        somatic_json TEXT NOT NULL DEFAULT '[]',
        acoustic_json TEXT NOT NULL DEFAULT '{"arousal": 0.5, "valence": 0.5}',
        is_locked INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    );
    """)

    # Ensure cssrs_json exists if table already existed
    try:
        if client.is_postgres:
            client.execute_write("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS cssrs_json TEXT NOT NULL DEFAULT '{}';")
        else:
            client.execute_write("ALTER TABLE sessions ADD COLUMN cssrs_json TEXT NOT NULL DEFAULT '{}';")
    except Exception:
        pass

    # Identity Vault Table (Secure Domain - Zero-Knowledge Isolated Encrypted PII)
    # Notice: NO FOREIGN KEY constraint linking to sessions table! Linked solely via session_id_hash.
    client.execute_write("""
    CREATE TABLE IF NOT EXISTS identity_vault (
        vault_id TEXT PRIMARY KEY,
        session_id_hash TEXT UNIQUE NOT NULL,
        masked_emirates_id TEXT NOT NULL,
        insurance_carrier TEXT NOT NULL,
        insurance_member_id TEXT NOT NULL,
        encrypted_pii_blob TEXT NOT NULL,
        bound_at REAL NOT NULL
    );
    """)

    # Clinical Reports Table (Encrypted FHIR Bundles & Ephemeral Access Tokens)
    client.execute_write("""
    CREATE TABLE IF NOT EXISTS clinical_reports (
        report_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        access_token TEXT UNIQUE NOT NULL,
        token_expires_at REAL NOT NULL,
        encrypted_fhir_bundle TEXT NOT NULL,
        consent_status TEXT NOT NULL DEFAULT 'PENDING'
    );
    """)

    # Scheduled Milestones Table (Longitudinal Reminder & Audit Scheduler)
    client.execute_write("""
    CREATE TABLE IF NOT EXISTS scheduled_milestones (
        milestone_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        target_timestamp REAL NOT NULL,
        touchpoint_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING',
        payload_json TEXT NOT NULL
    );
    """)


def save_session_state(state: Dict[str, Any], is_locked: Optional[bool] = None) -> None:
    """Insert or update session state in the sessions table."""
    client = _db
    session_id = state["session_id"]
    existing = get_session_by_id(session_id)
    
    phq9_json = json.dumps(state.get("phq9_slots", {}))
    gad7_json = json.dumps(state.get("gad7_slots", {}))
    cssrs_json = json.dumps(state.get("cssrs_slots", {}))
    somatic_json = json.dumps(state.get("somatic_flags", []))
    acoustic_json = json.dumps({
        "arousal": state.get("vocal_arousal", 0.5),
        "valence": state.get("vocal_valence", 0.5),
        "affect_discrepancy": state.get("affect_discrepancy", False)
    })
    
    lock_val = 1 if (is_locked or state.get("active_mode") == "EMERGENCY_LOCK" or state.get("cssrs_critical")) else 0

    if existing:
        lock_val = 1 if (existing.get("is_locked") or lock_val == 1) else 0
        try:
            client.execute_write("""
            UPDATE sessions SET
                language = %s,
                active_mode = %s,
                triage_tier = %s,
                care_modality = %s,
                phq9_json = %s,
                gad7_json = %s,
                cssrs_json = %s,
                somatic_json = %s,
                acoustic_json = %s,
                is_locked = %s
            WHERE session_id = %s;
            """, (
                state.get("language", "ar-AE"),
                state.get("active_mode", "CONVERSATIONAL"),
                state.get("triage_tier", "TIER_1_SELF_CARE"),
                state.get("care_modality", "CONVERSATIONAL"),
                phq9_json,
                gad7_json,
                cssrs_json,
                somatic_json,
                acoustic_json,
                lock_val,
                session_id
            ))
        except Exception:
            client.execute_write("""
            UPDATE sessions SET
                language = %s,
                active_mode = %s,
                triage_tier = %s,
                care_modality = %s,
                phq9_json = %s,
                gad7_json = %s,
                somatic_json = %s,
                acoustic_json = %s,
                is_locked = %s
            WHERE session_id = %s;
            """, (
                state.get("language", "ar-AE"),
                state.get("active_mode", "CONVERSATIONAL"),
                state.get("triage_tier", "TIER_1_SELF_CARE"),
                state.get("care_modality", "CONVERSATIONAL"),
                phq9_json,
                gad7_json,
                somatic_json,
                acoustic_json,
                lock_val,
                session_id
            ))
    else:
        created_at = time.time()
        try:
            client.execute_write("""
            INSERT INTO sessions (
                session_id, language, active_mode, triage_tier, care_modality,
                phq9_json, gad7_json, cssrs_json, somatic_json, acoustic_json, is_locked, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                session_id,
                state.get("language", "ar-AE"),
                state.get("active_mode", "CONVERSATIONAL"),
                state.get("triage_tier", "TIER_1_SELF_CARE"),
                state.get("care_modality", "CONVERSATIONAL"),
                phq9_json,
                gad7_json,
                cssrs_json,
                somatic_json,
                acoustic_json,
                lock_val,
                created_at
            ))
        except Exception:
            client.execute_write("""
            INSERT INTO sessions (
                session_id, language, active_mode, triage_tier, care_modality,
                phq9_json, gad7_json, somatic_json, acoustic_json, is_locked, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                session_id,
                state.get("language", "ar-AE"),
                state.get("active_mode", "CONVERSATIONAL"),
                state.get("triage_tier", "TIER_1_SELF_CARE"),
                state.get("care_modality", "CONVERSATIONAL"),
                phq9_json,
                gad7_json,
                somatic_json,
                acoustic_json,
                lock_val,
                created_at
            ))


def get_session_by_id(session_id: str) -> Optional[dict]:
    """Retrieve session record by session_id."""
    return _db.execute_read_one(
        "SELECT * FROM sessions WHERE session_id = %s;",
        (session_id,)
    )


def lock_session(session_id: str) -> None:
    """Hard emergency lock on a session."""
    _db.execute_write(
        "UPDATE sessions SET is_locked = 1, active_mode = 'EMERGENCY_LOCK' WHERE session_id = %s;",
        (session_id,)
    )


def bind_identity(
    session_id: str,
    full_name: str,
    emirates_id: str,
    insurance_carrier: str,
    insurance_member_id: str,
    phone: Optional[str] = None
) -> dict:
    """
    Masks Emirates ID to '784-****-*******-{last_digit}'.
    Encrypts PII into an isolated identity_vault table linked ONLY by SHA-256(session_id).
    """
    last_digit = emirates_id[-1] if emirates_id else "1"
    masked_eid = f"784-****-*******-{last_digit}"
    
    session_hash = hash_session_id(session_id)
    vault_id = f"vlt_{uuid.uuid4().hex[:12]}"
    
    pii_payload = json.dumps({
        "full_name": full_name,
        "emirates_id": emirates_id,
        "insurance_carrier": insurance_carrier,
        "insurance_member_id": insurance_member_id,
        "phone": phone or ""
    })
    encrypted_blob = encrypt_data(pii_payload)
    bound_at = time.time()

    # Check if already bound
    existing = _db.execute_read_one(
        "SELECT * FROM identity_vault WHERE session_id_hash = %s;",
        (session_hash,)
    )
    if existing:
        _db.execute_write("""
        UPDATE identity_vault SET
            masked_emirates_id = %s,
            insurance_carrier = %s,
            insurance_member_id = %s,
            encrypted_pii_blob = %s,
            bound_at = %s
        WHERE session_id_hash = %s;
        """, (
            masked_eid,
            insurance_carrier,
            insurance_member_id,
            encrypted_blob,
            bound_at,
            session_hash
        ))
    else:
        _db.execute_write("""
        INSERT INTO identity_vault (
            vault_id, session_id_hash, masked_emirates_id,
            insurance_carrier, insurance_member_id, encrypted_pii_blob, bound_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s);
        """, (
            vault_id,
            session_hash,
            masked_eid,
            insurance_carrier,
            insurance_member_id,
            encrypted_blob,
            bound_at
        ))

    return {
        "vault_id": vault_id,
        "masked_eid": masked_eid,
        "session_id_hash": session_hash
    }


def has_identity_bound(session_id: str) -> bool:
    """Check if the session has an active bound identity in the vault."""
    session_hash = hash_session_id(session_id)
    rec = _db.execute_read_one(
        "SELECT vault_id FROM identity_vault WHERE session_id_hash = %s;",
        (session_hash,)
    )
    return rec is not None


def get_identity_record(session_id: str) -> Optional[dict]:
    """Retrieve identity record by session_id hash."""
    session_hash = hash_session_id(session_id)
    return _db.execute_read_one(
        "SELECT * FROM identity_vault WHERE session_id_hash = %s;",
        (session_hash,)
    )


def create_report_token(session_id: str, fhir_bundle: dict, ttl_seconds: int = 3600) -> str:
    """
    Encrypts FHIR bundle, generates a 6-character access token (e.g. DOH-XXXX),
    and sets expiration to epoch + ttl_seconds (default 60 mins).
    """
    token_suffix = uuid.uuid4().hex[:4].upper()
    access_token = f"DOH-{token_suffix}"
    report_id = f"rep_{uuid.uuid4().hex[:12]}"
    token_expires_at = time.time() + ttl_seconds
    encrypted_fhir = encrypt_data(json.dumps(fhir_bundle))

    _db.execute_write("""
    INSERT INTO clinical_reports (
        report_id, session_id, access_token, token_expires_at, encrypted_fhir_bundle, consent_status
    ) VALUES (%s, %s, %s, %s, %s, 'PENDING');
    """, (
        report_id,
        session_id,
        access_token,
        token_expires_at,
        encrypted_fhir
    ))

    return access_token


def get_report_by_token(access_token: str) -> Optional[dict]:
    """Fetch report by access_token."""
    return _db.execute_read_one(
        "SELECT * FROM clinical_reports WHERE access_token = %s;",
        (access_token,)
    )


def authorize_report_token(access_token: str, granted: bool) -> Optional[dict]:
    """
    Patient decision on the consent challenge.
    If granted == True, unlocks decrypted FHIR bundle.
    """
    report = get_report_by_token(access_token)
    if not report:
        return None

    # Check token expiration
    if time.time() > float(report["token_expires_at"]):
        return None

    status = "GRANTED" if granted else "REVOKED"
    _db.execute_write(
        "UPDATE clinical_reports SET consent_status = %s WHERE access_token = %s;",
        (status, access_token)
    )

    if granted:
        decrypted_json = decrypt_data(report["encrypted_fhir_bundle"])
        return json.loads(decrypted_json)
    return None


def schedule_milestone(
    session_id: str,
    touchpoint_type: str,
    target_timestamp: float,
    payload: dict
) -> str:
    """Register milestone in scheduled_milestones table."""
    milestone_id = f"mil_{uuid.uuid4().hex[:12]}"
    _db.execute_write("""
    INSERT INTO scheduled_milestones (
        milestone_id, session_id, target_timestamp, touchpoint_type, status, payload_json
    ) VALUES (%s, %s, %s, %s, 'PENDING', %s);
    """, (
        milestone_id,
        session_id,
        target_timestamp,
        touchpoint_type,
        json.dumps(payload)
    ))
    return milestone_id


def get_milestones_for_session(session_id: str) -> List[dict]:
    """Retrieve all scheduled milestones for a given session."""
    return _db.execute_read_all(
        "SELECT * FROM scheduled_milestones WHERE session_id = %s ORDER BY target_timestamp ASC;",
        (session_id,)
    )


def get_all_recent_sessions(limit: int = 30) -> List[dict]:
    """Retrieve recent triage sessions for the clinician observer dashboard."""
    return _db.execute_read_all(
        "SELECT * FROM sessions ORDER BY created_at DESC LIMIT %s;",
        (limit,)
    )


def get_full_session_telemetry(session_id: str) -> Optional[dict]:
    """Fetch complete decoupled telemetry for clinician inspection."""
    session = get_session_by_id(session_id)
    if not session:
        return None

    identity = get_identity_record(session_id)
    reports = _db.execute_read_all(
        "SELECT report_id, access_token, token_expires_at, consent_status FROM clinical_reports WHERE session_id = %s;",
        (session_id,)
    )
    milestones = get_milestones_for_session(session_id)

    return {
        "session": session,
        "identity_vault": {
            "is_bound": identity is not None,
            "masked_eid": identity.get("masked_emirates_id") if identity else None,
            "carrier": identity.get("insurance_carrier") if identity else None,
            "bound_at": identity.get("bound_at") if identity else None,
            "session_id_hash": identity.get("session_id_hash") if identity else None
        },
        "clinical_reports": reports,
        "scheduled_milestones": milestones
    }

