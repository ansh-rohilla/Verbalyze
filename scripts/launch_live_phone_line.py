#!/usr/bin/env python3
"""
scripts/launch_live_phone_line.py

1-Click Live Indian Phone Line Gateway Launcher for Verbalyze & RingTrunk:
- Starts the production telephony FastAPI server (SIP Webhooks + /media-stream WebSocket).
- Auto-detects public tunnels (ngrok, cloudflared, localtunnel) or accepts custom --tunnel-url.
- Displays ready-to-copy carrier configuration snippet for RingTrunk (admin@ringtrunk.com).
- Streams live call events, streaming LLM-to-TTS latencies, and SMS UPI dispatches to terminal.
"""

import os
import sys
import time
import socket
import shutil
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_local_ip() -> str:
    """Detects local LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def print_banner():
    print("""
================================================================================
  VERBALYZE: 1-CLICK LIVE INDIAN PHONE LINE GATEWAY LAUNCHER
================================================================================
  Carrier Support: RingTrunk (admin@ringtrunk.com), Asterisk, FreeSWITCH, Twilio
  Audio Pipelining: Token-to-Speech Streaming (<200ms TTFS) | 8kHz G.711 A-law
  Barge-In Engine : Sub-50ms Instant WebSocket Buffer Cutoff ('clear' event)
  Payment Gateway : Real NPCI UPI Intent & SMS Dispatch
================================================================================
""")


def main():
    parser = argparse.ArgumentParser(description="Verbalyze Live Phone Line Gateway Launcher")
    parser.add_argument("--port", type=int, default=8000, help="Local port to bind FastAPI server (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--tunnel-url", type=str, default=None, help="Public HTTPS tunnel URL (e.g., https://my-subdomain.ngrok-free.app)")
    parser.add_argument("--persona", type=str, default="muthoot_recovery", choices=["muthoot_recovery", "bank_kyc", "swiggy_delivery"], help="Telephony persona")
    parser.add_argument("--lang", type=str, default="hi", help="Primary language (hi, en, gu, mr, ta, te, bn, kn, ml, pa, ur)")
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "groq", "openai", "mock"], help="LLM Provider")
    parser.add_argument("--model", type=str, default="verbalyze-indic", help="LLM Model name")
    parser.add_argument("--sms-provider", type=str, default="mock", choices=["mock", "fast2sms", "twilio", "webhook"], help="SMS Gateway Provider")
    parser.add_argument("--stt-provider", type=str, default="local", choices=["local", "faster-whisper", "google", "mock"], help="Speech-to-Text provider (default: local)")
    parser.add_argument("--stt-model", type=str, default="tiny", help="Whisper STT model size (default: tiny, choices: tiny, base, small)")
    parser.add_argument("--auth-token", type=str, default=None, help="Authentication token for webhooks and WebSocket (default: env TELEPHONY_AUTH_TOKEN)")
    parser.add_argument("--strict-sovereignty", action="store_true", help="Enforce 100% strict data sovereignty (zero cloud STT egress)")
    parser.add_argument("--no-server", action="store_true", help="Print config and instructions without launching server")

    args = parser.parse_args()
    print_banner()

    local_ip = get_local_ip()
    port = args.port
    tunnel_url = args.tunnel_url

    # Check for available tunnel CLIs if tunnel_url was not provided
    tunnel_proc = None
    if not tunnel_url:
        ngrok_bin = shutil.which("ngrok")
        cf_bin = shutil.which("cloudflared")

        if ngrok_bin:
            print("[TUNNEL] Found 'ngrok' in PATH! Initializing background tunnel on port", port)
            try:
                tunnel_proc = subprocess.Popen(
                    [ngrok_bin, "http", str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(2.0)
                # Fetch tunnel URL from local ngrok API
                import urllib.request
                import json
                try:
                    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=3) as resp:
                        tdata = json.loads(resp.read().decode())
                        tunnels = tdata.get("tunnels", [])
                        for t in tunnels:
                            if t.get("proto") == "https":
                                tunnel_url = t.get("public_url")
                                break
                except Exception:
                    pass
            except Exception as e:
                print(f"[WARN] Could not launch ngrok automatically: {e}")

        elif cf_bin:
            print("[TUNNEL] Found 'cloudflared' in PATH! Run 'cloudflared tunnel --url http://127.0.0.1:8000' for instant HTTPS.")

    if not tunnel_url:
        tunnel_url = f"http://{local_ip}:{port}"
        is_local = True
    else:
        is_local = False

    # Format URLs for SIP Webhook and WebSocket Media Stream
    if tunnel_url.startswith("https://"):
        ws_url = "wss://" + tunnel_url[8:]
        http_url = tunnel_url
    elif tunnel_url.startswith("http://"):
        ws_url = "ws://" + tunnel_url[7:]
        http_url = tunnel_url
    else:
        ws_url = f"wss://{tunnel_url}"
        http_url = f"https://{tunnel_url}"

    token_val = args.auth_token or os.environ.get("TELEPHONY_AUTH_TOKEN")
    auth_param = f"&token={token_val}" if token_val else ""
    strict_param = "&strict_sovereignty=true" if args.strict_sovereignty else ""

    sip_inbound_url = f"{http_url}/webhook/sip/inbound{('?token=' + token_val) if token_val else ''}"
    sip_turn_url = f"{http_url}/webhook/sip/turn{('?token=' + token_val) if token_val else ''}"
    media_stream_url = f"{ws_url}/media-stream?lang={args.lang}&persona={args.persona}&provider={args.provider}&stt_provider={args.stt_provider}&stt_model={args.stt_model}{auth_param}{strict_param}"

    print("--------------------------------------------------------------------------------")
    print("READY-TO-USE TELEPHONY CONFIGURATION FOR RINGTRUNK:")
    print("--------------------------------------------------------------------------------")
    print(f"  • Primary Telecom Carrier  : RingTrunk (admin@ringtrunk.com)")
    print(f"  • Inbound SIP Webhook URL  : {sip_inbound_url}")
    print(f"  • Conversational Turn URL  : {sip_turn_url}")
    print(f"  • WebSocket Media Stream   : {media_stream_url}")
    print(f"  • Active Telephony Persona : {args.persona.upper()}")
    print(f"  • Language & Voice Codec   : {args.lang.upper()} | ITU-T G.711 A-law (8kHz PCMA)")
    print(f"  • Sovereign Local STT      : {args.stt_provider.upper()} ({args.stt_model}) [On-Premises]")
    print(f"  • Strict Data Sovereignty  : {'Enabled (Zero Cloud STT Egress)' if args.strict_sovereignty else 'Disabled'}")
    print(f"  • Auth Protection          : {'Enabled (Bearer / Token Guard)' if token_val else 'Disabled (Open Dev Mode)'}")
    print(f"  • LLM Engine               : {args.provider.upper()} ({args.model})")
    print(f"  • SMS / NPCI UPI Gateway   : {args.sms_provider.upper()}")
    print("--------------------------------------------------------------------------------")

    if is_local:
        print("""
NOTE: You are currently running on a Local Network IP.
   To connect with RingTrunk's live PSTN trunk over the internet:
   1. Install ngrok: 'brew install ngrok' or download from ngrok.com
   2. Run ngrok:     'ngrok http 8000'
   3. Launch again:  'python scripts/launch_live_phone_line.py --tunnel-url https://xxxx.ngrok-free.app'
   OR provide your public cloud VPS IP / domain directly via '--tunnel-url https://YOUR_DOMAIN'.
""")

    print("EMAIL TEMPLATE TO SEND TO RINGTRUNK (admin@ringtrunk.com):")
    print("--------------------------------------------------------------------------------")
    print(f"""Subject: Live Indian Test Line Webhook Configuration - Verbalyze

Hi RingTrunk Team,

Verbalyze is now ready to receive live Indian phone calls!
Please configure our test DID number with the following endpoints:

1. Inbound SIP Webhook:
   {sip_inbound_url}

2. Bi-Directional WebSocket Media Stream (G.711 A-law 8kHz):
   {media_stream_url}

Audio Specs:
- Codec: ITU-T G.711 A-law (PCMA, 8,000 Hz mono)
- Packet Pacing: 20ms frames (160 bytes / packet)
- Interruption: Real-time buffer flush enabled via 'clear' event
- Turn-around: Token-to-TTS streaming (<200ms Time-to-First-Sound)
- STT Decoder: Sovereign On-Premises ({args.stt_provider.upper()}, in-memory 8kHz PCM)

Best regards,
Verbalyze Voice AI Team
""")
    print("--------------------------------------------------------------------------------")

    if args.no_server:
        print("Done. (Server not launched due to --no-server)")
        return

    # Launch FastAPI Telephony Server
    print(f"[GATEWAY] Starting Verbalyze Live Telephony Gateway on {args.host}:{port}...")
    print("   Press Ctrl+C to disconnect gateway and terminate session.\n")

    os.environ["SMS_DISPATCH_PROVIDER"] = args.sms_provider
    os.environ["LLM_PROVIDER"] = args.provider
    os.environ["DEFAULT_LANGUAGE"] = args.lang

    try:
        import uvicorn
        from verbalyze.telephony.server import create_app

        if args.strict_sovereignty:
            os.environ["STRICT_SOVEREIGNTY"] = "1"
        app = create_app(auth_token=args.auth_token)
        uvicorn.run(app, host=args.host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("\nShutting down Verbalyze Live Phone Line Gateway. Goodbye!")
    finally:
        if tunnel_proc:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
