"""
verbalyze/telephony/server.py

FastAPI telephony server providing SIP webhook connectors for Exotel and Twilio.
Enables real-time outbound call triggering and live voicebot turn-taking over SIP trunks.
"""

import os
import uuid
import asyncio
import base64
import json
import time
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
from verbalyze.agent.lid_engine import (
    LanguageIdentificationGate,
    LanguageIDResult,
)
from verbalyze.telephony.supervisor import SupervisorManager
from verbalyze.telephony.browser_gateway import BrowserAudioSession
from verbalyze.telephony.templates import SUPERVISOR_DASHBOARD_HTML, BROWSER_CLIENT_HTML
from verbalyze.telephony.whatsapp_gateway import WhatsAppGateway
from verbalyze.telephony.settlement_engine import SettlementLedger, SettlementStatus
from verbalyze.telephony.payment_webhooks import (
    verify_razorpay_signature,
    verify_cashfree_signature,
    verify_upi_webhook_signature,
    parse_razorpay_webhook,
    parse_cashfree_webhook,
    parse_upi_callback,
)
from verbalyze.telephony.call_recorder import DualChannelCallRecorder
from verbalyze.telephony.compliance_qa import (
    ComplianceQAEngine,
    ComplianceQARegistry,
    QAScorecard,
    ComplianceStatus,
    CRMNotes,
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
    lid_gate = LanguageIdentificationGate()
    supervisor_manager = SupervisorManager()
    whatsapp_gateway = WhatsAppGateway()
    settlement_ledger = SettlementLedger(
        supervisor_manager=supervisor_manager,
        whatsapp_gateway=whatsapp_gateway,
    )
    compliance_qa_registry = ComplianceQARegistry()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "active_calls": len(active_calls),
            "active_campaigns": len(active_campaigns),
            "supervisor_active_calls": len(supervisor_manager.active_calls),
            "settled_transactions": len([t for t in settlement_ledger.transactions.values() if t.status == SettlementStatus.SETTLED]),
            "audited_calls": len(compliance_qa_registry.scorecards),
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
                "sentiment": step_res.get("sentiment"),
                "language_info": step_res.get("language_info"),
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
            "sentiment": step_res.get("sentiment"),
            "language_info": step_res.get("language_info"),
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
            supervisor_manager=supervisor_manager,
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

    @app.post("/telephony/lid")
    async def analyze_lid_endpoint(request: Request):
        """
        Real-time multi-modal Language Identification (LID) & code-switching endpoint.
        Analyzes customer text and/or base64-encoded PCM audio for Indic language & script detection.
        """
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        data = await request.json()
        text = str(data.get("text", "")).strip()
        current_lang = str(data.get("current_language", "hi"))
        audio_b64 = data.get("audio_base64")
        pcm_bytes = None
        if audio_b64:
            try:
                pcm_bytes = base64.b64decode(audio_b64)
            except Exception:
                pass

        result = lid_gate.identify(
            transcript=text,
            pcm_bytes=pcm_bytes,
            current_language=current_lang,
        )
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

    # --------------------------------------------------------------------------
    # SUPERVISOR LIVE OBSERVABILITY & WHISPER COACHING
    # --------------------------------------------------------------------------

    @app.get("/telephony/supervisor/dashboard", response_class=HTMLResponse)
    async def supervisor_dashboard_ui():
        """Serves the real-time supervisor observability and live monitoring console."""
        return HTMLResponse(content=SUPERVISOR_DASHBOARD_HTML)

    @app.get("/telephony/supervisor/calls")
    async def list_active_supervisor_calls(request: Request):
        """Returns JSON snapshot of all active calls monitored by supervisors."""
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)
        return JSONResponse({
            "calls": supervisor_manager.list_calls(),
            "summary": supervisor_manager.get_fleet_summary()
        })

    @app.get("/telephony/supervisor/summary")
    async def get_supervisor_summary(request: Request):
        """Returns aggregated fleet health metrics across active calls."""
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)
        return JSONResponse(supervisor_manager.get_fleet_summary())

    @app.post("/telephony/supervisor/whisper")
    async def inject_supervisor_whisper_api(request: Request):
        """Injects private supervisor coaching directive into active call context."""
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)
        data = await request.json()
        call_id = str(data.get("call_id", "")).strip()
        text = str(data.get("text") or data.get("whisper", "")).strip()
        supervisor_id = str(data.get("supervisor_id", "supervisor_admin"))
        if not call_id or not text:
            return JSONResponse({"error": "Missing call_id or text parameter."}, status_code=400)
        success = supervisor_manager.inject_whisper(call_id, text, supervisor_id=supervisor_id)
        if not success:
            return JSONResponse({"error": f"Call {call_id} not found in active calls."}, status_code=404)
        return JSONResponse({"success": True, "call_id": call_id, "whisper": text})

    @app.post("/telephony/supervisor/takeover")
    async def takeover_call_api(request: Request):
        """Triggers immediate supervisor takeover and warm transfer for an active call."""
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)
        data = await request.json()
        call_id = str(data.get("call_id", "")).strip()
        supervisor_id = str(data.get("supervisor_id", "supervisor_admin"))
        target_sip = data.get("target_sip")
        reason = str(data.get("reason", "supervisor_manual_takeover"))
        res = supervisor_manager.takeover_call(call_id, supervisor_id=supervisor_id, target_sip=target_sip, reason=reason)
        return JSONResponse(res, status_code=200 if res.get("success") else 404)

    @app.post("/telephony/supervisor/barge-in")
    async def barge_in_api(request: Request):
        """Cuts agent audio speech immediately via supervisor command."""
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)
        data = await request.json()
        call_id = str(data.get("call_id", "")).strip()
        supervisor_id = str(data.get("supervisor_id", "supervisor_admin"))
        success = supervisor_manager.barge_in(call_id, supervisor_id=supervisor_id)
        return JSONResponse({"success": success, "call_id": call_id})

    @app.websocket("/telephony/supervisor/stream")
    async def supervisor_stream_endpoint(websocket: WebSocket, token: Optional[str] = None):
        """Real-time pub/sub event stream for supervisor dashboards."""
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
        queue = supervisor_manager.subscribe()
        try:
            init_payload = {
                "event": "initial_state",
                "summary": supervisor_manager.get_fleet_summary(),
                "calls": supervisor_manager.list_calls()
            }
            await websocket.send_text(json.dumps(init_payload))
            while True:
                msg = await queue.get()
                await websocket.send_text(msg)
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            supervisor_manager.unsubscribe(queue)

    # --------------------------------------------------------------------------
    # IN-BROWSER FULL-DUPLEX AUDIO GATEWAY
    # --------------------------------------------------------------------------

    @app.get("/telephony/browser-client", response_class=HTMLResponse)
    async def browser_client_ui():
        """Interactive in-browser telephony client for direct mic testing."""
        return HTMLResponse(content=BROWSER_CLIENT_HTML)

    @app.websocket("/telephony/browser/stream")
    async def browser_stream_endpoint(
        websocket: WebSocket,
        session_id: Optional[str] = None,
        lang: str = "hi",
        persona: str = "muthoot_recovery",
        provider: str = "ollama",
        model: Optional[str] = None,
        sample_rate: int = 16000,
        caller_phone: Optional[str] = None,
        stt_provider: str = "local",
        stt_model: str = "tiny"
    ):
        """Bi-directional full-duplex audio stream for in-browser callers."""
        await websocket.accept()
        call_id = session_id or f"web_{uuid.uuid4().hex[:8]}"
        session = BrowserAudioSession(
            websocket=websocket,
            session_id=call_id,
            language=lang,
            persona=persona,
            llm_provider=provider,
            model_name=model,
            sample_rate=sample_rate,
            caller_phone=caller_phone or "+919876543210",
            supervisor_manager=supervisor_manager,
            stt_provider=stt_provider,
            stt_model=stt_model
        )
        await session.run()

    # --------------------------------------------------------------------------
    # WHATSAPP BUSINESS API & RCS GATEWAY ENDPOINTS
    # --------------------------------------------------------------------------

    @app.get("/webhook/whatsapp")
    async def whatsapp_verify_challenge(request: Request):
        """
        Meta WhatsApp Webhook subscription verification endpoint.
        Validates hub.mode, hub.verify_token, and echoes back hub.challenge.
        """
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")

        if mode and token and challenge:
            verified_challenge = whatsapp_gateway.verify_webhook_challenge(mode, token, challenge)
            if verified_challenge:
                return Response(content=verified_challenge, media_type="text/plain", status_code=200)
            return Response(content="Forbidden: Invalid verification token", status_code=403)
        return Response(content="Bad Request: Missing parameters", status_code=400)

    @app.post("/webhook/whatsapp")
    async def whatsapp_incoming_webhook(request: Request):
        """
        Meta WhatsApp Business webhook listener.
        Processes message status delivery receipts and interactive Quick Reply button clicks.
        """
        raw_body = await request.body()
        sig_header = request.headers.get("x-hub-signature-256")
        if sig_header:
            app_secret = os.getenv("WHATSAPP_APP_SECRET") or os.getenv("WHATSAPP_API_TOKEN")
            if app_secret and not whatsapp_gateway.verify_payload_signature(raw_body, sig_header, app_secret):
                return JSONResponse({"status": "error", "message": "Invalid signature"}, status_code=401)

        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            return JSONResponse({"status": "error", "message": "Invalid JSON payload"}, status_code=400)

        event = whatsapp_gateway.handle_inbound_webhook(payload)

        # If button click was a callback request or dispute, broadcast to supervisor
        if event.get("event_type") == "button_click":
            button_id = event.get("button_id", "")
            sender_phone = event.get("sender_phone", "")
            if "CALLBACK" in button_id.upper():
                supervisor_manager._broadcast({
                    "event": "whatsapp_callback_requested",
                    "sender_phone": PIIRedactor.mask_phone(sender_phone),
                    "button_id": button_id,
                    "timestamp": time.time(),
                })
            elif "DISPUTE" in button_id.upper():
                supervisor_manager._broadcast({
                    "event": "whatsapp_dispute_raised",
                    "sender_phone": PIIRedactor.mask_phone(sender_phone),
                    "button_id": button_id,
                    "timestamp": time.time(),
                })

        return JSONResponse({"status": "success", "event": event})

    @app.post("/webhook/whatsapp/send")
    async def whatsapp_send_template(request: Request):
        """
        Dispatches an interactive WhatsApp payment or recovery notice to a customer.
        Requires valid authentication token if enabled.
        """
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        phone = payload.get("phone") or payload.get("phone_number")
        if not phone:
            return JSONResponse({"error": "Missing recipient phone number"}, status_code=400)

        customer_name = payload.get("customer_name", "Customer")
        loan_id = payload.get("loan_id", "MUTH-8921")
        amount = float(payload.get("amount", 5420.0))
        language = payload.get("language", "hi")
        overdue_days = int(payload.get("overdue_days", 14))

        result = whatsapp_gateway.dispatch_payment_message(
            phone_number=phone,
            customer_name=customer_name,
            loan_id=loan_id,
            amount=amount,
            overdue_days=overdue_days,
            language=language,
        )
        return JSONResponse(result)

    # --------------------------------------------------------------------------
    # SETTLEMENT ENGINE & NPCI UPI RECONCILIATION ENDPOINTS
    # --------------------------------------------------------------------------

    @app.post("/settlement/order")
    async def create_settlement_order(request: Request):
        """
        Creates a new payment settlement order in the ledger.
        Requires authentication guard verification.
        """
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        loan_id = payload.get("loan_id")
        phone = payload.get("phone") or payload.get("customer_phone")
        amount = payload.get("amount")

        if not loan_id or not phone or amount is None:
            return JSONResponse({"error": "Missing loan_id, phone, or amount"}, status_code=400)

        customer_name = payload.get("customer_name", "Borrower")
        gateway = payload.get("gateway", "UPI_INTENT")
        metadata = payload.get("metadata", {})

        txn = settlement_ledger.create_order(
            loan_id=str(loan_id),
            amount=float(amount),
            customer_phone=str(phone),
            customer_name=str(customer_name),
            gateway=str(gateway),
            metadata=metadata,
        )
        return JSONResponse({"status": "created", "transaction": txn.to_dict(mask_pii=True)})

    @app.get("/settlement/transactions")
    def list_settlement_transactions(request: Request, status: Optional[str] = None):
        """
        Lists all settlement orders and payments tracked by the ledger.
        Requires authentication guard verification.
        """
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        txns = list(settlement_ledger.transactions.values())
        if status:
            txns = [t for t in txns if t.status.value.lower() == status.lower()]

        return JSONResponse({
            "count": len(txns),
            "transactions": [t.to_dict(mask_pii=True) for t in txns]
        })

    @app.get("/settlement/receipt/{transaction_id}")
    def get_settlement_receipt_pdf(request: Request, transaction_id: str):
        """
        Generates and serves official digital payment receipt PDF directly from memory.
        No disk clutter, zero temporary storage.
        """
        pdf_bytes = settlement_ledger.generate_pdf_receipt_bytes(transaction_id)
        if not pdf_bytes:
            return JSONResponse({
                "error": f"Settled receipt for transaction {transaction_id} not found."
            }, status_code=404)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="receipt_{transaction_id}.pdf"'}
        )

    # --------------------------------------------------------------------------
    # PAYMENT GATEWAY WEBHOOK INGESTION (Razorpay / Cashfree / BharatPe UPI)
    # --------------------------------------------------------------------------

    @app.post("/webhook/payment/razorpay")
    async def razorpay_payment_webhook(request: Request):
        """
        Razorpay payment webhook receiver.
        Performs constant-time HMAC-SHA256 signature verification and auto-reconciles settled payments.
        """
        raw_body = await request.body()
        sig = request.headers.get("x-razorpay-signature", "")
        secret = os.getenv("RAZORPAY_WEBHOOK_SECRET", "razorpay_secret_key_123")

        if not verify_razorpay_signature(raw_body, sig, secret):
            return JSONResponse({"status": "error", "message": "Invalid Razorpay HMAC-SHA256 signature"}, status_code=400)

        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            return JSONResponse({"status": "error", "message": "Malformed JSON payload"}, status_code=400)

        parsed = parse_razorpay_webhook(payload)
        order_id = parsed.get("order_id") or parsed.get("notes", {}).get("loan_id") or parsed.get("notes", {}).get("transaction_id")
        payment_id = parsed.get("payment_id", f"pay_rzp_{uuid.uuid4().hex[:8]}")

        # Reconcile payment in ledger
        success, msg, txn = settlement_ledger.reconcile_payment(
            transaction_id=order_id,
            gateway_payment_id=payment_id,
            signature=sig,
            raw_payload=raw_body,
            verify_sig=False,  # Already verified above
            gateway="RAZORPAY",
        )

        # Halt retries across all active campaigns
        if txn:
            for dialer in active_campaigns.values():
                dialer.mark_lead_settled(txn.loan_id)
                dialer.mark_lead_settled(txn.customer_phone_raw)

        return JSONResponse({
            "status": "reconciled" if success else "failed",
            "message": msg,
            "transaction_id": txn.transaction_id if txn else None,
            "receipt_number": txn.receipt_number if txn else None,
        }, status_code=200 if success else 404)

    @app.post("/webhook/payment/cashfree")
    async def cashfree_payment_webhook(request: Request):
        """
        Cashfree payment webhook receiver.
        Validates timestamped HMAC-SHA256 signature and auto-reconciles settled loans.
        """
        raw_body = await request.body()
        sig = request.headers.get("x-webhook-signature", "")
        timestamp = request.headers.get("x-webhook-timestamp", "")
        secret = os.getenv("CASHFREE_WEBHOOK_SECRET", "cashfree_secret_key_123")

        if not verify_cashfree_signature(raw_body, sig, timestamp, secret):
            return JSONResponse({"status": "error", "message": "Invalid Cashfree HMAC-SHA256 signature"}, status_code=400)

        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            return JSONResponse({"status": "error", "message": "Malformed JSON payload"}, status_code=400)

        parsed = parse_cashfree_webhook(payload)
        order_id = parsed.get("order_id")
        payment_id = parsed.get("payment_id", f"pay_cf_{uuid.uuid4().hex[:8]}")

        success, msg, txn = settlement_ledger.reconcile_payment(
            transaction_id=order_id,
            gateway_payment_id=payment_id,
            signature=sig,
            raw_payload=raw_body,
            verify_sig=False,
            gateway="CASHFREE",
        )

        if txn:
            for dialer in active_campaigns.values():
                dialer.mark_lead_settled(txn.loan_id)
                dialer.mark_lead_settled(txn.customer_phone_raw)

        return JSONResponse({
            "status": "reconciled" if success else "failed",
            "message": msg,
            "transaction_id": txn.transaction_id if txn else None,
            "receipt_number": txn.receipt_number if txn else None,
        }, status_code=200 if success else 404)

    @app.post("/webhook/payment/upi")
    async def upi_callback_webhook(request: Request):
        """
        Direct NPCI / BharatPe / UPI callback webhook receiver.
        Validates HMAC signature and reconciles settlement against loan.
        """
        raw_body = await request.body()
        sig = request.headers.get("x-webhook-signature", "")
        secret = os.getenv("UPI_WEBHOOK_SECRET", "settlement_webhook_secret_key_123")

        if not verify_upi_webhook_signature(raw_body, sig, secret):
            return JSONResponse({"status": "error", "message": "Invalid UPI webhook signature"}, status_code=400)

        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            return JSONResponse({"status": "error", "message": "Malformed JSON payload"}, status_code=400)

        parsed = parse_upi_callback(payload)
        order_id = parsed.get("order_id")
        upi_ref = parsed.get("upi_ref_id") or parsed.get("gateway_payment_id", f"upi_ref_{uuid.uuid4().hex[:8]}")

        success, msg, txn = settlement_ledger.reconcile_payment(
            transaction_id=order_id,
            gateway_payment_id=upi_ref,
            signature=sig,
            raw_payload=raw_body,
            verify_sig=False,
            gateway="UPI_INTENT",
        )

        if txn:
            for dialer in active_campaigns.values():
                dialer.mark_lead_settled(txn.loan_id)
                dialer.mark_lead_settled(txn.customer_phone_raw)

        return JSONResponse({
            "status": "reconciled" if success else "failed",
            "message": msg,
            "transaction_id": txn.transaction_id if txn else None,
            "receipt_number": txn.receipt_number if txn else None,
        }, status_code=200 if success else 404)

    # --------------------------------------------------------------------------
    # REGULATORY COMPLIANCE QA & DUAL-CHANNEL CALL AUDITING (RBI / TRAI)
    # --------------------------------------------------------------------------

    @app.post("/telephony/qa/evaluate")
    async def evaluate_call_compliance(request: Request):
        """
        Runs post-call 4-pillar compliance audit on a conversation transcript.
        Requires authentication guard verification.
        """
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        call_id = payload.get("call_id") or f"CALL_{uuid.uuid4().hex[:8].upper()}"
        turns = payload.get("turns", [])
        loan_id = payload.get("loan_id", "MUTH-8921")
        phone = payload.get("phone") or payload.get("caller_phone") or "+919876543210"
        customer_name = payload.get("customer_name", "Borrower")
        duration = float(payload.get("duration_sec", 0.0))
        language = payload.get("language", "hi")
        start_time = payload.get("start_timestamp")

        scorecard = ComplianceQAEngine.evaluate_call(
            call_id=call_id,
            turns=turns,
            loan_id=loan_id,
            caller_phone=phone,
            customer_name=customer_name,
            call_start_timestamp=start_time,
            audio_duration_sec=duration,
            language=language,
        )

        compliance_qa_registry.register_audit(scorecard)
        return JSONResponse({"status": "evaluated", "scorecard": scorecard.to_dict(mask_pii=True)})

    @app.get("/telephony/qa/audits")
    def list_compliance_audits(request: Request, status: Optional[str] = None):
        """
        Lists all audited calls and compliance scorecards.
        Requires authentication guard verification.
        """
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        audits = compliance_qa_registry.list_audits()
        if status:
            audits = [a for a in audits if a.get("status", "").lower() == status.lower()]

        return JSONResponse({
            "count": len(audits),
            "audits": audits,
        })

    @app.get("/telephony/qa/audit/{call_id}")
    def get_compliance_audit(request: Request, call_id: str):
        """
        Retrieves full QA scorecard, CRM notes, and violation details for a specific call.
        Requires authentication guard verification.
        """
        if not _verify_request(request):
            return JSONResponse({"error": "Unauthorized: Invalid or missing authentication token."}, status_code=401)

        scorecard = compliance_qa_registry.get_scorecard(call_id)
        if not scorecard:
            return JSONResponse({"error": f"Audit record for call '{call_id}' not found."}, status_code=404)

        return JSONResponse(scorecard.to_dict(mask_pii=True))

    @app.get("/telephony/qa/certificate/{call_id}")
    def get_compliance_certificate_pdf(request: Request, call_id: str):
        """
        Generates and serves official RBI Compliance Audit Certificate PDF directly from memory.
        Zero disk storage overhead.
        """
        pdf_bytes = compliance_qa_registry.generate_certificate_pdf_bytes(call_id)
        if not pdf_bytes:
            return JSONResponse({"error": f"Certificate for call '{call_id}' not found."}, status_code=404)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="compliance_certificate_{call_id}.pdf"'}
        )

    @app.get("/telephony/qa/recording/{call_id}")
    def get_call_recording_wav(request: Request, call_id: str):
        """
        Streams official dual-channel stereo WAV audio recording (Channel 0: Customer, Channel 1: Agent).
        Served directly from memory.
        """
        wav_bytes = compliance_qa_registry.get_recording(call_id)
        if not wav_bytes:
            return JSONResponse({"error": f"Audio recording for call '{call_id}' not found."}, status_code=404)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="call_recording_{call_id}.wav"'}
        )

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
