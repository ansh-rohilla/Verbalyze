"""
verbalyze/telephony/server.py

FastAPI telephony server providing SIP webhook connectors for Exotel and Twilio.
Enables real-time outbound call triggering and live voicebot turn-taking over SIP trunks.
"""

import os
from typing import Dict, Any, Optional
from verbalyze.agent.voice_bot import VoiceAgent
from verbalyze.telephony.media_stream import MediaStreamSession

# Try importing FastAPI
try:
    from fastapi import FastAPI, Request, Response, Form, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def create_app() -> Any:
    """Creates the FastAPI telephony application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required to run the telephony server. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Verbalyze Telephony Voicebot Server",
        description="Production webhook bridge connecting Exotel & Twilio SIP trunks to Verbalyze Indic Voice SLMs.",
        version="0.2.0"
    )

    # In-memory call session store: call_sid -> VoiceAgent
    active_calls: Dict[str, VoiceAgent] = {}

    @app.get("/health")
    def health():
        return {"status": "ok", "active_calls": len(active_calls), "engine": "Verbalyze Telephony v0.2.0"}

    @app.post("/webhook/twilio/voice")
    async def twilio_incoming_call(request: Request):
        """Initial webhook when an outbound/inbound call connects on Twilio."""
        form = await request.form()
        call_sid = str(form.get("CallSid", "call_mock"))
        lang = str(form.get("lang", "hi"))
        stream_mode = str(form.get("stream", "false")).lower() == "true" or "stream" in str(request.query_params).lower()

        if stream_mode:
            # Connect call directly to real-time bi-directional WebSocket media stream
            host = request.url.netloc
            ws_protocol = "wss" if request.url.scheme == "https" else "ws"
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_protocol}://{host}/media-stream" />
    </Connect>
</Response>"""
            return Response(content=twiml, media_type="application/xml")

        # Instantiate agent for this specific telephone call
        agent = VoiceAgent(language=lang, voice_enabled=False)
        active_calls[call_sid] = agent

        greeting = (
            "नमस्कार, क्या मेरी बात मिस्टर शर्मा से हो रही है? मैं मुथूट फिनकॉर्प से बोल रही हूँ।"
            if lang == "hi" else
            "Hello, am I speaking with Mr. Sharma? I am calling from Muthoot Fincorp regarding your EMI."
        )

        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/webhook/twilio/turn?call_sid={call_sid}" language="{lang}-IN" timeout="3">
        <Say language="{lang}-IN">{greeting}</Say>
    </Gather>
</Response>"""
        return Response(content=twiml, media_type="application/xml")

    @app.post("/webhook/twilio/turn")
    async def twilio_call_turn(request: Request, call_sid: Optional[str] = None):
        """Processes each spoken turn from the customer via Twilio Speech Recognition."""
        form = await request.form()
        call_sid = call_sid or str(form.get("CallSid", "call_mock"))
        speech_result = str(form.get("SpeechResult", "")).strip()

        agent = active_calls.get(call_sid)
        if not agent:
            agent = VoiceAgent(language="hi", voice_enabled=False)
            active_calls[call_sid] = agent

        if not speech_result:
            rep = "क्षमा करें, क्या आप दोहरा सकते हैं?"
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/webhook/twilio/turn?call_sid={call_sid}" timeout="3">
        <Say>{rep}</Say>
    </Gather>
</Response>"""
            return Response(content=twiml, media_type="application/xml")

        # Step the agent
        step_res = agent.step(speech_result)
        agent_reply = step_res["text"]

        if step_res["terminated"]:
            # Hang up the call
            active_calls.pop(call_sid, None)
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>{agent_reply}</Say>
    <Hangup/>
</Response>"""
        else:
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/webhook/twilio/turn?call_sid={call_sid}" timeout="3">
        <Say>{agent_reply}</Say>
    </Gather>
</Response>"""

        return Response(content=twiml, media_type="application/xml")

    @app.post("/call/simulate")
    async def simulate_call_api(customer_input: str, call_id: str = "test_call", lang: str = "hi"):
        """REST API testing endpoint for programmatic call turn-taking."""
        agent = active_calls.get(call_id)
        if not agent:
            agent = VoiceAgent(language=lang, voice_enabled=False)
            active_calls[call_id] = agent

        res = agent.step(customer_input)
        if res["terminated"]:
            active_calls.pop(call_id, None)

        return {
            "call_id": call_id,
            "agent_response": res["text"],
            "tool_event": res["tool_event"],
            "call_active": not res["terminated"]
        }

    # --------------------------------------------------------------------------
    # UNMETERED SIP TRUNK ENDPOINTS (RingTrunk.com / Asterisk / FreeSWITCH)
    # Allows flat-rate channel SIP trunking without per-minute carrier bills.
    # --------------------------------------------------------------------------

    @app.post("/webhook/sip/inbound")
    async def sip_inbound_call(request: Request):
        """
        Generic SIP inbound call webhook compatible with unmetered SIP trunks (RingTrunk, Asterisk, FreeSWITCH).
        Accepts JSON or form data with Call-ID, Caller, and Dialed Number.
        """
        data: Dict[str, Any] = {}
        try:
            data = await request.json()
        except Exception:
            form = await request.form()
            data = dict(form)

        call_id = str(data.get("call_id") or request.headers.get("x-call-id") or request.headers.get("call-id") or "sip_call_001")
        lang = str(data.get("lang") or "hi")
        persona = str(data.get("persona") or "muthoot_recovery")
        provider = str(data.get("provider") or "ollama")

        # Initialize VoiceAgent for this SIP session
        agent = VoiceAgent(language=lang, persona=persona, llm_provider=provider, voice_enabled=True)
        active_calls[call_id] = agent

        greeting = agent.get_initial_greeting()
        audio_path = None
        if agent.audio_engine:
            audio_path = agent.audio_engine.synthesize(greeting)

        host = request.url.netloc
        ws_protocol = "wss" if request.url.scheme == "https" else "ws"
        return JSONResponse({
            "status": "connected",
            "trunk_type": "unmetered_sip",
            "provider": "RingTrunk / Standard SIP",
            "call_id": call_id,
            "greeting_text": greeting,
            "media_stream_ws": f"{ws_protocol}://{host}/media-stream?lang={lang}&persona={persona}&provider={provider}",
            "audio_url": audio_path,
            "action": "play_and_listen"
        })

    @app.post("/webhook/sip/turn")
    async def sip_call_turn(request: Request):
        """
        Processes conversational spoken turns over an unmetered SIP trunk stream.
        """
        data: Dict[str, Any] = {}
        try:
            data = await request.json()
        except Exception:
            form = await request.form()
            data = dict(form)

        call_id = str(data.get("call_id") or "sip_call_001")
        customer_utterance = str(data.get("transcript") or data.get("customer_input") or "").strip()

        agent = active_calls.get(call_id)
        if not agent:
            agent = VoiceAgent(language="hi", llm_provider="ollama", voice_enabled=True)
            active_calls[call_id] = agent

        if not customer_utterance:
            return JSONResponse({
                "call_id": call_id,
                "agent_response": "क्षमा करें, क्या आप दोहरा सकते हैं?",
                "tool_event": None,
                "hangup": False
            })

        step_res = agent.step(customer_utterance)
        terminated = step_res["terminated"]

        if terminated:
            active_calls.pop(call_id, None)

        return JSONResponse({
            "call_id": call_id,
            "agent_response": step_res["text"],
            "audio_url": step_res.get("audio_path"),
            "tool_event": step_res.get("tool_event"),
            "quality_report": step_res.get("quality_report").to_dict() if step_res.get("quality_report") else None,
            "hangup": terminated,
            "action": "hangup" if terminated else "play_and_listen"
        })

    # --------------------------------------------------------------------------
    # BI-DIRECTIONAL WEBSOCKET MEDIA STREAM (RingTrunk / Twilio / Asterisk)
    # --------------------------------------------------------------------------

    @app.websocket("/media-stream")
    @app.websocket("/webhook/sip/media")
    async def media_stream_endpoint(
        websocket: WebSocket,
        lang: str = "hi",
        persona: str = "muthoot_recovery",
        provider: str = "ollama",
        model: Optional[str] = None,
        codec: str = "audio/x-alaw"
    ):
        """
        Real-time bi-directional audio WebSocket endpoint for live telephony trunks
        (RingTrunk, Twilio Media Streams, Asterisk AudioSocket, FreeSWITCH).
        Streams 20ms G.711 A-law/mu-law audio packets with sub-50ms live barge-in interruption.
        """
        await websocket.accept()
        session = MediaStreamSession(
            websocket=websocket,
            language=lang,
            persona=persona,
            llm_provider=provider,
            model_name=model,
            codec=codec
        )
        await session.run()

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
