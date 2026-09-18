"""
verbalyze/telephony/server.py

FastAPI telephony server providing SIP webhook connectors for Exotel and Twilio.
Enables real-time outbound call triggering and live voicebot turn-taking over SIP trunks.
"""

import os
import uuid
import asyncio
import base64
from typing import Dict, Any, Optional
from verbalyze.agent.voice_bot import VoiceAgent
from verbalyze.telephony.media_stream import MediaStreamSession
from verbalyze.security import verify_auth_token, PIIRedactor
from verbalyze.campaign import (
    CampaignDialer,
    CampaignConfig,
    AMDClassifier,
)
from verbalyze.agent.sentiment import (
    UnifiedSentimentEngine,
    SentimentResult,
    DisputeType,
    SentimentCategory,
)
from verbalyze.telephony.transfer import (
    SIPTransferDispatcher,
    TransferContext,
)

# Try importing FastAPI
try:
    from fastapi import FastAPI, Request, Response, Form, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def create_app(auth_token: Optional[str] = None) -> Any:
    """Creates the FastAPI telephony application with end-to-end security guards."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required to run the telephony server. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Verbalyze Telephony Voicebot Server",
        description="Production webhook bridge connecting Exotel & Twilio SIP trunks to Verbalyze Indic Voice SLMs.",
        version="0.3.0"
    )

    # Authentication token: passed directly, or via environment TELEPHONY_AUTH_TOKEN / VERBALYZE_API_KEY
    expected_token = (
        auth_token
        or os.environ.get("TELEPHONY_AUTH_TOKEN")
        or os.environ.get("VERBALYZE_API_KEY")
    )

    def _verify_request(request: Request) -> bool:
        """Validates inbound HTTP requests against configured authentication token."""
        if not expected_token:
            return True
        # 1. Bearer token in Authorization header
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if verify_auth_token(token, expected_token):
                return True
        # 2. X-Verbalyze-Token or X-Auth-Token
        x_token = request.headers.get("x-verbalyze-token") or request.headers.get("x-auth-token")
        if x_token and verify_auth_token(x_token, expected_token):
            return True
        # 3. Query parameter ?token=...
        q_token = request.query_params.get("token")
        if q_token and verify_auth_token(q_token, expected_token):
            return True
        return False

    # In-memory call session store: call_sid -> VoiceAgent
    active_calls: Dict[str, VoiceAgent] = {}
    active_campaigns: Dict[str, CampaignDialer] = {}
    amd_engine = AMDClassifier()
    sentiment_engine = UnifiedSentimentEngine()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "active_calls": len(active_calls),
            "active_campaigns": len(active_campaigns),
            "auth_enabled": bool(expected_token),
            "engine": "Verbalyze Telephony v0.2.0"
        }

    @app.post("/webhook/twilio/voice")
    async def twilio_incoming_call(request: Request):
        """Initial webhook when an outbound/inbound call connects on Twilio."""
        if not _verify_request(request):
            return Response(content='<Response><Reject reason="rejected"/></Response>', media_type="application/xml", status_code=401)

        form = await request.form()
        call_sid = str(form.get("CallSid", "call_mock"))
        lang = str(form.get("lang", "hi"))
        stream_mode = str(form.get("stream", "false")).lower() == "true" or "stream" in str(request.query_params).lower()

        if stream_mode:
            # Connect call directly to real-time bi-directional WebSocket media stream
            host = request.url.netloc
            ws_protocol = "wss" if request.url.scheme == "https" else "ws"
            token_query = f"?token={expected_token}" if expected_token else ""
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_protocol}://{host}/media-stream{token_query}" />
    </Connect>
</Response>"""
            return Response(content=twiml, media_type="application/xml")

        from_phone = str(form.get("From") or form.get("Caller") or "")

        # Instantiate agent for this specific telephone call
        agent = VoiceAgent(language=lang, voice_enabled=False, caller_phone=from_phone)
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
        if not _verify_request(request):
            return Response(content='<Response><Reject reason="rejected"/></Response>', media_type="application/xml", status_code=401)

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
        tool_data = step_res.get("tool_data") or {}

        if tool_data.get("action") == "transfer":
            active_calls.pop(call_sid, None)
            transfer_ctx = TransferContext(
                call_id=call_sid,
                caller_phone=agent.caller_phone or "",
                loan_id="MUTH-8921",
                amount_due=5420.0,
                agitation_score=step_res.get("sentiment", {}).get("composite_agitation", 0.8),
                dispute_type=tool_data.get("reason", "DISPUTE"),
                briefing_summary=tool_data.get("summary", "Customer transfer escalation."),
                target_department=tool_data.get("department", "supervisor"),
            )
            twiml = SIPTransferDispatcher.build_twiml_dial_transfer(
                context=transfer_ctx,
                caller_id=agent.caller_phone,
                language=agent.language,
            )
            return Response(content=twiml, media_type="application/xml")

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
    async def simulate_call_api(request: Request, customer_input: str, call_id: str = "test_call", lang: str = "hi"):
        """REST API testing endpoint for programmatic call turn-taking."""
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

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
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

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
        strict_sovereignty = str(data.get("strict_sovereignty", "")).lower() in ("true", "1") or os.environ.get("STRICT_SOVEREIGNTY", "0") in ("1", "true")

        caller_phone = str(data.get("caller_phone") or data.get("from") or data.get("caller_id") or request.headers.get("x-caller-phone") or "")

        # Initialize VoiceAgent for this SIP session
        agent = VoiceAgent(language=lang, persona=persona, llm_provider=provider, voice_enabled=True, caller_phone=caller_phone)
        active_calls[call_id] = agent

        greeting = agent.get_initial_greeting()
        audio_path = None
        if agent.audio_engine:
            audio_path = agent.audio_engine.synthesize(greeting)

        host = request.url.netloc
        ws_protocol = "wss" if request.url.scheme == "https" else "ws"
        token_param = f"&token={expected_token}" if expected_token else ""
        strict_param = "&strict_sovereignty=true" if strict_sovereignty else ""
        return JSONResponse({
            "status": "connected",
            "trunk_type": "unmetered_sip",
            "provider": "RingTrunk / Standard SIP",
            "call_id": call_id,
            "greeting_text": greeting,
            "media_stream_ws": f"{ws_protocol}://{host}/media-stream?lang={lang}&persona={persona}&provider={provider}{token_param}{strict_param}",
            "audio_url": audio_path,
            "action": "play_and_listen"
        })

    @app.post("/webhook/sip/turn")
    async def sip_call_turn(request: Request):
        """
        Processes conversational spoken turns over an unmetered SIP trunk stream.
        """
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

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
        tool_data = step_res.get("tool_data") or {}

        if tool_data.get("action") == "transfer":
            active_calls.pop(call_id, None)
            transfer_ctx = TransferContext(
                call_id=call_id,
                caller_phone=agent.caller_phone or "",
                loan_id="MUTH-8921",
                amount_due=5420.0,
                agitation_score=step_res.get("sentiment", {}).get("composite_agitation", 0.8),
                dispute_type=tool_data.get("reason", "DISPUTE"),
                briefing_summary=tool_data.get("summary", "Customer transfer escalation."),
                target_department=tool_data.get("department", "supervisor"),
            )
            sip_refer = SIPTransferDispatcher.build_sip_refer(context=transfer_ctx)
            return JSONResponse({
                "call_id": call_id,
                "agent_response": step_res["text"],
                "audio_url": step_res.get("audio_path"),
                "tool_event": step_res.get("tool_event"),
                "quality_report": step_res.get("quality_report").to_dict() if step_res.get("quality_report") else None,
                "hangup": True,
                "action": "transfer",
                "sip_refer": sip_refer,
                "transfer_context": transfer_ctx.to_dict(mask_pii=True),
            })

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
        token: Optional[str] = None,
        lang: str = "hi",
        persona: str = "muthoot_recovery",
        provider: str = "ollama",
        model: Optional[str] = None,
        codec: str = "audio/x-alaw",
        caller_phone: Optional[str] = None,
        stt_provider: str = "local",
        stt_model: str = "tiny",
        strict_sovereignty: bool = False
    ):
        """
        Real-time bi-directional audio WebSocket endpoint with timing-attack resistant authentication guard.
        Streams 20ms G.711 A-law/mu-law audio packets with sub-50ms live barge-in interruption.
        """
        # Verify Token before accepting handshake
        ws_token = token or websocket.query_params.get("token")
        if not ws_token:
            auth_header = websocket.headers.get("authorization", "")
            if auth_header.lower().startswith("bearer "):
                ws_token = auth_header[7:].strip()
            else:
                ws_token = websocket.headers.get("x-verbalyze-token") or websocket.headers.get("x-auth-token")

        if expected_token and not verify_auth_token(ws_token, expected_token):
            await websocket.close(code=1008, reason="Policy Violation: Unauthorized")
            return

        await websocket.accept()
        session = MediaStreamSession(
            websocket=websocket,
            language=lang,
            persona=persona,
            llm_provider=provider,
            model_name=model,
            codec=codec,
            caller_phone=caller_phone,
            stt_provider=stt_provider,
            stt_model=stt_model,
            strict_sovereignty=strict_sovereignty
        )
        await session.run()

    # --------------------------------------------------------------------------
    # OUTBOUND CAMPAIGN & AMD REST ENDPOINTS
    # --------------------------------------------------------------------------

    @app.post("/campaign/start")
    async def start_campaign(request: Request):
        """
        Initiates an automated outbound batch campaign across concurrent channels.
        Requires authentication guard verification.
        """
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        payload = {}
        try:
            payload = await request.json()
        except Exception:
            form = await request.form()
            payload = dict(form)

        campaign_id = str(payload.get("campaign_id") or f"CAMP_{uuid.uuid4().hex[:8].upper()}")
        persona = str(payload.get("persona") or "muthoot_recovery")
        language = str(payload.get("language") or payload.get("lang") or "hi")
        provider = str(payload.get("provider") or payload.get("llm_provider") or "ollama")
        model = payload.get("model") or payload.get("model_name")
        channels = int(payload.get("channels") or payload.get("max_concurrent_channels") or 5)
        enforce_hours = str(payload.get("enforce_calling_hours", "true")).lower() in ("true", "1")
        enforce_dnd = str(payload.get("enforce_dnd", "true")).lower() in ("true", "1")

        config = CampaignConfig(
            campaign_id=campaign_id,
            campaign_name=str(payload.get("campaign_name") or f"Campaign {campaign_id}"),
            persona=persona,
            language=language,
            llm_provider=provider,
            model_name=model,
            max_concurrent_channels=channels,
            enforce_trai_calling_hours=enforce_hours,
            enforce_dnd_check=enforce_dnd,
        )

        dialer = CampaignDialer(config=config)

        # Ingest leads
        raw_leads = payload.get("leads", [])
        if isinstance(raw_leads, str):
            accepted, rejected, errors = dialer.ingest_csv(raw_leads)
        elif isinstance(raw_leads, list):
            accepted, rejected, errors = dialer.ingest_leads_from_list(raw_leads)
        else:
            accepted, rejected, errors = 0, 0, ["No valid leads provided"]

        if accepted == 0:
            return JSONResponse({
                "status": "rejected",
                "campaign_id": campaign_id,
                "error": "No valid leads accepted for dialing.",
                "rejected_count": rejected,
                "rejection_errors": errors[:5],
            }, status_code=400)

        active_campaigns[campaign_id] = dialer

        # Launch campaign in background task
        asyncio.create_task(dialer.run_campaign())

        return JSONResponse({
            "status": "started",
            "campaign_id": campaign_id,
            "accepted_leads": accepted,
            "rejected_leads": rejected,
            "max_channels": channels,
            "enforce_trai_hours": enforce_hours,
        })

    @app.get("/campaign/status/{campaign_id}")
    def get_campaign_status(request: Request, campaign_id: str):
        """Returns real-time progress, dispositions, and statistics for a campaign."""
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        dialer = active_campaigns.get(campaign_id)
        if not dialer:
            return JSONResponse({"error": f"Campaign '{campaign_id}' not found."}, status_code=404)

        return JSONResponse(dialer.summary.to_dict())

    @app.get("/campaign/cdr/{campaign_id}")
    def get_campaign_cdrs(request: Request, campaign_id: str, mask_pii: bool = True):
        """Exports PII-sanitized Call Detail Records (CDRs) for an outbound campaign."""
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        dialer = active_campaigns.get(campaign_id)
        if not dialer:
            return JSONResponse({"error": f"Campaign '{campaign_id}' not found."}, status_code=404)

        cdrs = [c.to_dict(mask_pii=mask_pii) for c in dialer.cdrs]
        return JSONResponse({
            "campaign_id": campaign_id,
            "count": len(cdrs),
            "pii_masked": mask_pii,
            "call_detail_records": cdrs,
        })

    @app.post("/telephony/amd")
    async def analyze_amd_endpoint(request: Request):
        """Standalone Answering Machine Detection endpoint for telephony webhook hooks."""
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        data = await request.json()
        transcript = str(data.get("transcript", ""))
        duration = float(data.get("speech_duration_sec", 1.0))
        silence_ratio = float(data.get("silence_ratio", 0.0))
        silence_after = float(data.get("silence_after_burst_sec", 0.0))

        result = amd_engine.classify(
            audio_duration_sec=duration,
            transcript=transcript,
            silence_after_burst_sec=silence_after,
            silence_ratio=silence_ratio,
        )
        return JSONResponse(result.to_dict())

    # --------------------------------------------------------------------------
    # REAL-TIME SENTIMENT & SIP REFER WARM TRANSFER ENDPOINTS
    # --------------------------------------------------------------------------

    @app.post("/telephony/sentiment")
    async def analyze_sentiment_endpoint(request: Request):
        """
        Real-time acoustic and lexical sentiment analysis endpoint.
        Analyzes customer text and/or base64-encoded PCM audio for agitation and disputes.
        """
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        data = await request.json()
        text = str(data.get("text", "")).strip()
        audio_b64 = data.get("audio_base64")
        pcm_bytes = None
        if audio_b64:
            try:
                pcm_bytes = base64.b64decode(audio_b64)
            except Exception:
                pass

        result = sentiment_engine.analyze(text=text, pcm_bytes=pcm_bytes)
        return JSONResponse(result.to_dict())

    @app.post("/telephony/transfer")
    async def dispatch_transfer_endpoint(request: Request):
        """
        Dispatches a carrier warm transfer directive (SIP REFER, Twilio TwiML, or WebSocket frame).
        Injects X-Verbalyze-Context metadata header briefing the receiving agent.
        """
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        data = await request.json()
        call_id = str(data.get("call_id") or f"CALL_{uuid.uuid4().hex[:8].upper()}")
        caller_phone = str(data.get("caller_phone") or "")
        loan_id = str(data.get("loan_id") or "MUTH-8921")
        amount_due = float(data.get("amount_due", 0.0))
        agitation = float(data.get("agitation_score", 0.0))
        dispute_type = str(data.get("dispute_type", "NONE"))
        briefing = str(data.get("briefing_summary", "Warm transfer initiated."))
        target_dept = str(data.get("target_department", "supervisor"))
        target_uri = data.get("target_uri")
        fmt = str(data.get("format", "sip_refer")).lower()
        lang = str(data.get("language", "hi"))

        context = TransferContext(
            call_id=call_id,
            caller_phone=caller_phone,
            loan_id=loan_id,
            amount_due=amount_due,
            agitation_score=agitation,
            dispute_type=dispute_type,
            briefing_summary=briefing,
            target_department=target_dept,
        )

        if fmt == "twiml":
            twiml = SIPTransferDispatcher.build_twiml_dial_transfer(
                target=target_uri,
                context=context,
                caller_id=caller_phone,
                language=lang,
            )
            return Response(content=twiml, media_type="application/xml")
        elif fmt == "websocket":
            event = SIPTransferDispatcher.build_websocket_transfer_event(
                target_uri=target_uri,
                context=context,
            )
            return JSONResponse(event)
        else:
            refer_headers = SIPTransferDispatcher.build_sip_refer(
                target_uri=target_uri,
                context=context,
            )
            return JSONResponse({
                "status": "transfer_initiated",
                "format": "sip_refer",
                "call_id": call_id,
                "sip_refer_headers": refer_headers,
                "context": context.to_dict(mask_pii=True),
            })

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
