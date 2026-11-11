"""
test_finalized_freeze_and_observer.py - Verification script for:
1. Chat session freeze when agentic decisions are finalized (preventing creeping messages).
2. Slot indicators and live coverage counter payload in observer stream.
"""
import asyncio
import json
import websockets

async def test_session_freeze_and_slots():
    session_id = "test_freeze_verify_001"
    uri_chat = f"ws://localhost:8000/ws/triage/{session_id}"
    uri_obs = f"ws://localhost:8000/ws/observer/{session_id}"

    async with websockets.connect(uri_obs) as ws_obs:
        # Initial snapshot from observer
        obs_init = json.loads(await ws_obs.recv())
        assert obs_init["type"] == "OBSERVER_CONNECTED"
        print("[PASS] Observer connected successfully.")

        async with websockets.connect(uri_chat) as ws_chat:
            # Turn 1: Cardiac red flag triggers Level 3 (TRIGGER_JIT_IDENTITY_GATE)
            packet = {
                "transcript": "I feel severe chest tightness and my heart is racing",
                "arousal": 0.85,
                "valence": 0.2,
                "language": "en-US"
            }
            await ws_chat.send(json.dumps(packet))
            chat_reply = json.loads(await ws_chat.recv())
            
            assert chat_reply["type"] == "AGENT_TURN"
            action = chat_reply.get("dispatched_action")
            assert action is not None, "Expected an action to be dispatched"
            assert action["action_type"] == "TRIGGER_JIT_IDENTITY_GATE"
            assert chat_reply["slots"] is not None
            assert len(chat_reply["slots"]["somatic_flags"]) >= 1
            print(f"[PASS] Action Dispatched: {action['action_type']} with slots payload.")

            # Observer should receive OBSERVER_TURN_UPDATE with slots
            obs_update = json.loads(await ws_obs.recv())
            assert obs_update["type"] == "OBSERVER_TURN_UPDATE"
            assert "slots" in obs_update
            assert obs_update["slots"]["somatic_flags"] is not None
            print("[PASS] Observer received OBSERVER_TURN_UPDATE with slot indicators.")

            # Turn 2: Attempt to send another message AFTER decision was finalized
            creeping_packet = {
                "transcript": "Hello are you still there?",
                "arousal": 0.5,
                "valence": 0.5,
                "language": "en-US"
            }
            await ws_chat.send(json.dumps(creeping_packet))
            frozen_reply = json.loads(await ws_chat.recv())
            
            # Must be rejected with SESSION_FINALIZED, preventing LLM re-invocation
            assert frozen_reply["type"] == "SESSION_FINALIZED"
            assert "Clinical triage assessment has been finalized" in frozen_reply["reply_text"]
            print(f"[PASS] Creeping message blocked by backend: {frozen_reply['type']} - '{frozen_reply['reply_text']}'")

if __name__ == "__main__":
    asyncio.run(test_session_freeze_and_slots())
    print("\nALL FREEZE AND OBSERVER VERIFICATION CHECKS PASSED!")
