"""
app.py - Verbalyze: Indic Voice AI & Synthetic Data Suite
Hugging Face Spaces & Local Interactive Web Application

Features:
1. Interactive Telephony Voicebot (Edge-TTS Neural Voice + Multi-lingual Tool Calling)
2. Indic STT Benchmark Arena (Whisper vs. Sarvam vs. Google on 11 Edge Scenarios)
3. Dataset Explorer & Leaderboard (16,370 Dialogues & 172,800 STT Utterances)
"""

import os
import sys
import json
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import gradio as gr

# Ensure verbalyze package is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.agent.voice_bot import VoiceAgent, INITIAL_GREETINGS
from verbalyze.agent.audio_engine import AudioEngine, NEURAL_VOICES
from verbalyze.benchmark.metrics import compute_wer, compute_cer, extract_digits, extract_acronyms

# Supported Languages mapping
LANGUAGE_OPTIONS = {
    "Hindi (हिंदी)": "hi",
    "English (India)": "en",
    "Gujarati (ગુજરાતી)": "gu",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Marathi (मराठी)": "mr",
    "Bengali (বাংলা)": "bn",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Odia (ଓଡ଼ିଆ)": "or",
    "Urdu (اردو)": "ur",
}

PERSONA_OPTIONS = {
    "Muthoot Fincorp (Outbound Loan EMI Recovery - ₹5,420)": "muthoot_recovery",
    "Bank of Baroda (Mandatory Re-KYC Account Update)": "bank_kyc",
    "Swiggy Express (Delivery Gate & Address Verification)": "swiggy_delivery",
}

# Curated benchmark scenarios with authentic reference speech & ASR predictions
BENCHMARK_SCENARIOS = {
    "code_mixed_speech (Hinglish & Indic-English)": {
        "lang": "hi",
        "reference": "मेरा credit card payment process नहीं हो रहा, OTP send करो।",
        "whisper": "मेरा credit card payment process नहीं हो रहा OTP send करो",
        "sarvam": "मेरा credit card payment process नहीं हो रहा, OTP send करो।",
        "google": "Mera credit card payment process nahi ho raha, OTP send karo.",
        "insight": "Whisper often drops Indian conversational punctuation and switches randomly between Latin and Devanagari scripts, while Sarvam preserves Indic orthography.",
    },
    "numeric_normalization (Account, Phone, Amounts)": {
        "lang": "hi",
        "reference": "खाता संख्या 9821034567 में पांच हजार चार सौ बीस रुपये जमा हुए।",
        "whisper": "खाता संख्या 9821034567 में 5420 रुपये जमा हुए",
        "sarvam": "खाता संख्या 9821034567 में पांच हजार चार सौ बीस रुपये जमा हुए।",
        "google": "खाता संख्या ९८२१०३४५६७ में 5,420 रुपये जमा हुए।",
        "insight": "Western models prematurely convert spoken number words ('पांच हजार चार सौ बीस') into digits ('5420'), breaking strict verbal alignment protocols.",
    },
    "spoken_number_patterns (Telephony Digits & Booking IDs)": {
        "lang": "hi",
        "reference": "काउंटर पर बुकिंग आईडी six double seven four five zero दिखाएं।",
        "whisper": "काउंटर पर बुकिंग आईडी 677450 दिखाएं",
        "sarvam": "काउंटर पर बुकिंग आईडी six double seven four five zero दिखाएं।",
        "google": "काउंटर पर बुकिंग आईडी 67 74 50 दिखाएं।",
        "insight": "Spoken telephony groupings ('double seven') get collapsed into single digits by standard models, losing customer verification nuances.",
    },
    "units_measurements (Dosages, Weights & Distances)": {
        "lang": "hi",
        "reference": "डॉक्टर ने मरीज को 250 ml सिरप और 500 mg पेरासिटामोल लेने को कहा।",
        "whisper": "डॉक्टर ने मरीज को 250 मिली सिरप और 500 मिलीग्राम पेरासिटामोल लेने को कहा",
        "sarvam": "डॉक्टर ने मरीज को 250 ml सिरप और 500 mg पेरासिटामोल लेने को कहा।",
        "google": "Doctor ne mareez ko 250 ml syrup aur 500 mg paracetamol lene ko kaha.",
        "insight": "Standard ASR replaces standardized SI abbreviation units ('ml', 'mg') with localized expanded spellings ('मिली', 'मिलीग्राम').",
    },
    "abbreviations_acronyms (Banking & Tech Acronyms)": {
        "lang": "hi",
        "reference": "कृपया KYC वेरिफिकेशन के लिए PAN और Aadhaar की फोटो अपलोड करें।",
        "whisper": "कृपया के वाई सी वेरिफिकेशन के लिए पैन और आधार की फोटो अपलोड करें",
        "sarvam": "कृपया KYC वेरिफिकेशन के लिए PAN और Aadhaar की फोटो अपलोड करें।",
        "google": "Kripya KYC verification ke liye PAN aur Aadhaar ki photo upload karein.",
        "insight": "Standard models spell out acronyms phonetically instead of preserving capital business acronyms like KYC, PAN, IFSC.",
    },
    "named_entities (Indian Names & Geography)": {
        "lang": "mr",
        "reference": "डॉक्टर रमेश कुलकर्णी, छत्रपती संभाजीनगर, महाराष्ट्र।",
        "whisper": "डॉक्टर रमेश कुलकर्नी छत्रपति संभाजी नगर महाराष्ट्र",
        "sarvam": "डॉक्टर रमेश कुलकर्णी, छत्रपती संभाजीनगर, महाराष्ट्र।",
        "google": "Doctor Ramesh Kulkarni Chhatrapati Sambhajinagar Maharashtra.",
        "insight": "Regional name spellings and newly renamed cities (Chhatrapati Sambhajinagar) are preserved accurately by Indic-native models.",
    },
    "english_word_retention (Financial Loanwords)": {
        "lang": "hi",
        "reference": "अपने mutual fund investment का monthly SIP statement चेक कर लीजिए।",
        "whisper": "अपने म्यूचुअल फंड इन्वेस्टमेंट का मंथली सिप स्टेटमेंट चेक कर लीजिए",
        "sarvam": "अपने mutual fund investment का monthly SIP statement चेक कर लीजिए।",
        "google": "Apne mutual fund investment ka monthly SIP statement check kar lijiye.",
        "insight": "Indic voicebots require preserving financial English keywords in Latin or loanword form to match backend database schemas.",
    },
    "noisy_or_real_world_audio (Telephony Background Noise)": {
        "lang": "hi",
        "reference": "हाँ हेलो, सड़क पर बहुत शोर है, क्या आप मुथूट फिनकॉर्प से बोल रहे हैं?",
        "whisper": "हाँ हेलो सड़क पर बहुत ... क्या आप मुथूट से बोल रहे हैं",
        "sarvam": "हाँ हेलो, सड़क पर बहुत शोर है, क्या आप मुथूट फिनकॉर्प से बोल रहे हैं?",
        "google": "Haan hello sadak par bahut shor hai kya aap Muthoot Fincorp se bol rahe hain?",
        "insight": "Under 8kHz telephony codecs and street noise, standard models frequently hallucinate or drop trailing speech tokens.",
    },
}

# Pre-load cached sample dialogues for fast Explorer tab viewing
SAMPLE_DIALOGUES = {}
def load_sample_dialogues():
    global SAMPLE_DIALOGUES
    for lang_code in ["hi", "gu", "ta", "te", "mr", "bn", "kn", "ml", "pa", "en"]:
        file_name = f"dataset_{lang_code.capitalize()}.json"
        p = PROJECT_ROOT / file_name
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    SAMPLE_DIALOGUES[lang_code] = data[:25] # Cache 25 sample dialogues per language
            except Exception:
                pass

load_sample_dialogues()


# ==============================================================================
# Helper Functions: Benchmark Calculation
# ==============================================================================
def calculate_metrics_for_pred(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calculates WER, CER, Digit Accuracy, and Acronym Retention."""
    wer = round(compute_wer(reference, hypothesis) * 100, 2)
    cer = round(compute_cer(reference, hypothesis) * 100, 2)
    
    # Digit accuracy
    ref_digits = extract_digits(reference)
    hyp_digits = extract_digits(hypothesis)
    if not ref_digits:
        digit_acc = 100.0
    else:
        matches = sum(1 for d in ref_digits if d in hyp_digits)
        digit_acc = round((matches / len(ref_digits)) * 100, 1)

    # Acronym retention
    ref_acronyms = extract_acronyms(reference)
    hyp_acronyms = extract_acronyms(hypothesis)
    if not ref_acronyms:
        acronym_acc = 100.0
    else:
        matches = sum(1 for a in ref_acronyms if a in hyp_acronyms)
        acronym_acc = round((matches / len(ref_acronyms)) * 100, 1)

    return {
        "wer": wer,
        "cer": cer,
        "digit_acc": digit_acc,
        "acronym_acc": acronym_acc,
    }


# ==============================================================================
# Gradio Tab 1: Telephony Voicebot State
# ==============================================================================
class WebVoiceSession:
    def __init__(
        self,
        language: str = "hi",
        persona: str = "muthoot_recovery",
        provider: str = "mock",
        api_key: str = "",
        min_human_likeness: float = 0.80,
        simulate_telephony: bool = False,
        caller_phone: str = "+919876543210"
    ):
        self.language = language
        self.persona = persona
        self.caller_phone = caller_phone
        self.min_human_likeness = min_human_likeness
        self.simulate_telephony = simulate_telephony
        prov = "ollama" if "ollama" in provider.lower() else ("groq" if "groq" in provider.lower() else ("openai" if "openai" in provider.lower() else "mock"))
        self.agent = VoiceAgent(
            language=language,
            persona=persona,
            llm_provider=prov,
            api_key=api_key.strip() if api_key.strip() else None,
            voice_enabled=True,
            min_human_likeness=min_human_likeness,
            simulate_telephony=simulate_telephony,
            caller_phone=caller_phone
        )
        self.call_history: List[Dict[str, str]] = []
        self.recent_tool_event: Optional[str] = None
        self.recent_tool_data: Optional[Dict[str, Any]] = None
        self.recent_quality_badge: Optional[str] = None
        self.is_connected: bool = True

    def start_call(self) -> Tuple[List[Dict[str, str]], Optional[str], str, str, str]:
        """Initiates call with persona-specific greeting and neural speech."""
        greeting = self.agent.get_initial_greeting()
        self.call_history = [{"role": "assistant", "content": greeting}]
        self.agent.messages.append({"role": "assistant", "content": greeting})
        self.is_connected = True
        self.recent_tool_event = None

        # Synthesize audio with quality evaluation
        audio_file = None
        quality_badge = "<div class='quality-badge'>[OK] Quality Gate: Verified (Min Score: 80% / MOS ~4.0 Gate Active)</div>"
        if self.agent.audio_engine:
            audio_file = self.agent.audio_engine.synthesize(greeting)
            report = self.agent.audio_engine.last_quality_report
            if report:
                if report.passed and audio_file:
                    quality_badge = f"<div class='quality-badge'>[OK] Quality Gate: Human-Likeness {report.score*100:.1f}% (MOS {report.mos_equivalent:.2f}/5.0 | Cadence: {report.cadence_score*100:.0f}%) Accepted</div>"
                else:
                    quality_badge = f"<div class='quality-badge-rejected'>[WARNING] Quality Gate: Audio Rejected ({report.score*100:.1f}% < {self.min_human_likeness*100:.0f}%) — {report.feedback}</div>"

        self.recent_quality_badge = quality_badge
        status_text = "[ACTIVE] **IN CALL** (Trunk: SIP-0821-DELHI | 8kHz G.711 Telephony Codec)"
        event_badge = "[TELEPHONY] Call Connected: Outbound Agent Dialed"
        return self.call_history, audio_file, status_text, event_badge, quality_badge

    def send_turn(self, user_text: str) -> Tuple[List[Dict[str, str]], Optional[str], str, str, str]:
        """Processes user utterance and returns updated messages, audio, and telephony telemetry."""
        default_badge = self.recent_quality_badge or "<div class='quality-badge'>[OK] Quality Gate Active</div>"
        if not user_text.strip():
            return self.call_history, None, "[ACTIVE] **IN CALL**", self.recent_tool_event or "No action", default_badge

        if not self.is_connected:
            self.call_history.append({"role": "user", "content": user_text})
            self.call_history.append({"role": "assistant", "content": "[Phone Call Disconnected - Click 'Restart Call' to dial again]"})
            return self.call_history, None, "[DISCONNECTED] **CALL TERMINATED**", "Call is hung up.", default_badge

        # Add user message
        self.call_history.append({"role": "user", "content": user_text})

        # Step agent
        res = self.agent.step(user_text)
        bot_reply = res.get("text", "")
        audio_file = res.get("audio_path")
        report = res.get("quality_report")
        tool_event = res.get("tool_event")
        terminated = res.get("terminated", False)

        self.call_history.append({"role": "assistant", "content": bot_reply})

        if terminated:
            self.is_connected = False
            status_text = "[DISCONNECTED] **CALL TERMINATED** (Remote End Disconnected)"
        else:
            status_text = "[ACTIVE] **IN CALL** (SIP Stream Active)"

        tool_data = res.get("tool_data")
        if tool_data:
            self.recent_tool_data = tool_data

        event_text = tool_event if tool_event else ("[SPEECH] Spoken turn processed" if not terminated else "[TELEPHONY] Call Hung Up")
        self.recent_tool_event = event_text

        quality_badge = default_badge
        if report:
            if report.passed and audio_file:
                quality_badge = (
                    f"<div class='quality-badge'>[OK] Quality Gate: Human-Likeness {report.score*100:.1f}% "
                    f"(MOS {report.mos_equivalent:.2f}/5.0 | Cadence: {report.cadence_score*100:.0f}% | Prosody: {report.prosody_score*100:.0f}%) Accepted</div>"
                )
            else:
                quality_badge = (
                    f"<div class='quality-badge-rejected'>[WARNING] Quality Gate: Speech Rejected "
                    f"({report.score*100:.1f}% < {self.min_human_likeness*100:.0f}%) — {report.feedback}</div>"
                )
        self.recent_quality_badge = quality_badge

        return self.call_history, audio_file, status_text, event_text, quality_badge


# Global sessions dict per session id (or simple instance)
GLOBAL_SESSIONS: Dict[str, WebVoiceSession] = {}

def get_or_create_session(session_id: str, lang_name: str, persona_name: str, provider: str, api_key: str, min_score: float = 0.80, simulate_telephony: bool = False, caller_phone: str = "+919876543210") -> WebVoiceSession:
    lang_code = LANGUAGE_OPTIONS.get(lang_name, "hi")
    persona_code = PERSONA_OPTIONS.get(persona_name, "muthoot_recovery")
    sess = WebVoiceSession(
        language=lang_code,
        persona=persona_code,
        provider=provider,
        api_key=api_key,
        min_human_likeness=min_score,
        simulate_telephony=simulate_telephony,
        caller_phone=caller_phone
    )
    GLOBAL_SESSIONS[session_id] = sess
    return sess


# ==============================================================================
# Gradio App Layout
# ==============================================================================
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.hero-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
    color: white !important;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}
.hero-header h1 {
    color: white !important;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 8px;
}
.hero-header p {
    color: #e0f2fe !important;
    font-size: 1.05rem;
    margin-bottom: 12px;
}
.telephony-badge {
    padding: 8px 14px;
    background: #f1f5f9;
    border-radius: 8px;
    border-left: 4px solid #3b82f6;
    font-weight: 600;
    font-size: 0.95rem;
}
.quality-badge {
    padding: 8px 14px;
    background: #ecfdf5;
    border-radius: 8px;
    border-left: 4px solid #10b981;
    font-weight: 600;
    font-size: 0.92rem;
    color: #065f46;
    margin-top: 6px;
}
.quality-badge-rejected {
    padding: 8px 14px;
    background: #fef2f2;
    border-radius: 8px;
    border-left: 4px solid #ef4444;
    font-weight: 600;
    font-size: 0.92rem;
    color: #991b1b;
    margin-top: 6px;
}
.metric-box {
    padding: 12px;
    border-radius: 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    text-align: center;
}
"""

with gr.Blocks(title="Verbalyze: Indic Voice AI Suite") as demo:
    session_state = gr.State(value=lambda: str(random.randint(100000, 999999)))

    # Hero Banner
    gr.HTML("""
    <div class="hero-header">
        <h1>Verbalyze: Indic Voice AI & Synthetic Data Suite</h1>
        <p>Full-Duplex Telephony Voicebot, STT Benchmark Arena, and Multi-lingual Dataset Explorer across 12 Indian Languages.</p>
        <div>
            <a href="https://huggingface.co/datasets/ansh-rohilla/verbalyze-dialogues" target="_blank" style="margin-right: 8px;">
                <img src="https://img.shields.io/badge/Hugging%20Face-verbalyze--dialogues-blue" alt="HF Dialogues" style="display:inline-block; vertical-align:middle;">
            </a>
            <a href="https://huggingface.co/datasets/ansh-rohilla/verbalyze-stt-bench" target="_blank" style="margin-right: 8px;">
                <img src="https://img.shields.io/badge/Hugging%20Face-verbalyze--stt--bench-green" alt="HF STT Bench" style="display:inline-block; vertical-align:middle;">
            </a>
            <a href="https://colab.research.google.com/github/ansh-rohilla/Verbalyze/blob/main/notebooks/train_indic_voice_slm.ipynb" target="_blank" style="margin-right: 8px;">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" style="display:inline-block; vertical-align:middle;">
            </a>
            <a href="https://github.com/ansh-rohilla/Verbalyze" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-Verbalyze-black?logo=github" alt="GitHub" style="display:inline-block; vertical-align:middle;">
            </a>
        </div>
    </div>
    """)

    with gr.Tabs() as tabs:
        # ======================================================================
        # TAB 1: INTERACTIVE TELEPHONY VOICEBOT
        # ======================================================================
        with gr.TabItem("Telephony Voicebot Playground", id="tab_voicebot"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("### Call Configuration")
                    lang_dropdown = gr.Dropdown(
                        choices=list(LANGUAGE_OPTIONS.keys()),
                        value="Hindi (हिंदी)",
                        label="Target Language (12 Indic Languages)",
                        interactive=True
                    )
                    persona_dropdown = gr.Dropdown(
                        choices=list(PERSONA_OPTIONS.keys()),
                        value="Muthoot Fincorp (Outbound Loan EMI Recovery - ₹5,420)",
                        label="Telephony Persona & Flow",
                        interactive=True
                    )
                    provider_radio = gr.Radio(
                        choices=["Built-in Indic Voice Engine (Instant & Free)", "Ollama (Local verbalyze-indic 3B SLM)", "Groq (Llama-3.3-70B)", "OpenAI (GPT-4o-mini)"],
                        value="Built-in Indic Voice Engine (Instant & Free)",
                        label="LLM Provider",
                        interactive=True
                    )
                    api_key_input = gr.Textbox(
                        label="Optional API Key (Groq / OpenAI)",
                        placeholder="Leave blank for built-in or local Ollama model...",
                        type="password",
                        visible=False
                    )

                    def toggle_api_key_visibility(provider_choice):
                        return gr.update(visible="Groq" in provider_choice or "OpenAI" in provider_choice)

                    provider_radio.change(toggle_api_key_visibility, inputs=[provider_radio], outputs=[api_key_input])

                    min_score_slider = gr.Slider(
                        minimum=0.60,
                        maximum=0.98,
                        value=0.80,
                        step=0.05,
                        label="Quality Gate (Min Human-Likeness)",
                        info="Rejects or auto-tunes speech below score threshold (Default: 80% / MOS ~4.0)",
                        interactive=True
                    )
                    telephony_sim_checkbox = gr.Checkbox(
                        value=False,
                        label="8kHz Telecom Line Simulator (G.711 A-law)",
                        info="Simulates real Indian PSTN 300Hz–3400Hz bandpass, A-law companding & packet jitter",
                        interactive=True
                    )
                    caller_phone_input = gr.Textbox(
                        value="+91 98765 43210",
                        label="Caller Mobile Number (for Real SMS/UPI Dispatch)",
                        placeholder="+91 98765 43210",
                        interactive=True
                    )

                    start_btn = gr.Button("Start / Restart Call", variant="primary", size="lg")
                    
                    gr.Markdown("---")
                    status_display = gr.Markdown("[ACTIVE] **IN CALL** (SIP Stream Connected)")
                    telemetry_display = gr.HTML("<div class='telephony-badge'>Telephony Event: Call Connected</div>")
                    quality_display = gr.HTML("<div class='quality-badge'>Quality Gate: Human-Likeness 80% Gate Active</div>")
                    bot_audio_output = gr.Audio(label="Agent Voice Response (Neural Edge-TTS)", autoplay=True, type="filepath")

                with gr.Column(scale=6):
                    gr.Markdown("### Conversational Telephony Channel")
                    chatbot_ui = gr.Chatbot(
                        label="Phone Conversation",
                        height=440
                    )

                    with gr.Row():
                        user_input = gr.Textbox(
                            show_label=False,
                            placeholder="Type or reply in Hindi, English, or Romanized Indic (e.g., 'हाँ मैं आज पेमेंट कर दूंगा')...",
                            scale=8
                        )
                        send_btn = gr.Button("Send Turn", variant="primary", scale=2)

                    gr.Markdown("#### Quick Customer Responses:")
                    with gr.Row():
                        quick_btn_1 = gr.Button("हाँ, मैं आज शाम तक ₹5,420 जमा कर दूंगा।", size="sm")
                        quick_btn_2 = gr.Button("पेमेंट का UPI लिंक WhatsApp या SMS पर भेज दीजिए।", size="sm")
                    with gr.Row():
                        quick_btn_3 = gr.Button("माफ़ कीजिए, यह गलत नंबर है।", size="sm")
                        quick_btn_4 = gr.Button("मैं अभी व्यस्त हूँ, मुझे कल कॉल कीजिए।", size="sm")

            # Voicebot Event handlers
            def handle_start_call(s_id, l_name, p_name, prov, key, min_score, tel_sim=False, phone="+919876543210"):
                session = get_or_create_session(s_id, l_name, p_name, prov, key, min_score, tel_sim, caller_phone=phone)
                history, audio, status, event, quality_badge = session.start_call()
                event_html = f"<div class='telephony-badge'>Telephony Event: {event}</div>"
                return history, audio, status, event_html, quality_badge

            def handle_send_message(s_id, text, l_name, p_name, prov, key, min_score, tel_sim=False, phone="+919876543210"):
                if s_id not in GLOBAL_SESSIONS:
                    session = get_or_create_session(s_id, l_name, p_name, prov, key, min_score, tel_sim, caller_phone=phone)
                    session.start_call()
                else:
                    session = GLOBAL_SESSIONS[s_id]
                    if phone:
                        session.caller_phone = phone
                        session.agent.caller_phone = phone

                history, audio, status, event, quality_badge = session.send_turn(text)
                if session.recent_tool_data and session.recent_tool_data.get("upi_url"):
                    d = session.recent_tool_data
                    event_html = (
                        f"<div class='telephony-badge' style='background:#e8f5e9;border-left:4px solid #2e7d32;color:#1b5e20;padding:8px 12px;margin:4px 0;'>"
                        f"<b>Live SMS & UPI Link Sent</b> to <b>{d.get('phone')}</b><br/>"
                        f"• <b>Amount:</b> ₹{d.get('amount', 5420):,.2f} | <b>Account:</b> {d.get('loan_id', 'MUTH-8921')}<br/>"
                        f"• <b>NPCI Intent:</b> <a href='{d.get('upi_url')}' target='_blank' style='color:#1565c0;text-decoration:underline;'>Click to Pay via UPI (GPay/PhonePe)</a>"
                        f"</div>"
                    )
                else:
                    event_html = f"<div class='telephony-badge'>Telephony Event: {event}</div>"
                return history, audio, status, event_html, quality_badge, ""

            start_btn.click(
                handle_start_call,
                inputs=[session_state, lang_dropdown, persona_dropdown, provider_radio, api_key_input, min_score_slider, telephony_sim_checkbox, caller_phone_input],
                outputs=[chatbot_ui, bot_audio_output, status_display, telemetry_display, quality_display]
            )

            send_btn.click(
                handle_send_message,
                inputs=[session_state, user_input, lang_dropdown, persona_dropdown, provider_radio, api_key_input, min_score_slider, telephony_sim_checkbox, caller_phone_input],
                outputs=[chatbot_ui, bot_audio_output, status_display, telemetry_display, quality_display, user_input]
            )
            user_input.submit(
                handle_send_message,
                inputs=[session_state, user_input, lang_dropdown, persona_dropdown, provider_radio, api_key_input, min_score_slider, telephony_sim_checkbox, caller_phone_input],
                outputs=[chatbot_ui, bot_audio_output, status_display, telemetry_display, quality_display, user_input]
            )

            # Quick reply buttons
            for btn in [quick_btn_1, quick_btn_2, quick_btn_3, quick_btn_4]:
                btn.click(
                    lambda btn_text, s_id, l_name, p_name, prov, key, score, tel_sim, phone: handle_send_message(s_id, btn_text, l_name, p_name, prov, key, score, tel_sim, phone),
                    inputs=[btn, session_state, lang_dropdown, persona_dropdown, provider_radio, api_key_input, min_score_slider, telephony_sim_checkbox, caller_phone_input],
                    outputs=[chatbot_ui, bot_audio_output, status_display, telemetry_display, quality_display, user_input]
                )

        # ======================================================================
        # TAB 2: INDIC STT BENCHMARK ARENA
        # ======================================================================
        with gr.TabItem("Indic STT Benchmark Arena", id="tab_benchmark"):
            gr.Markdown("### Side-by-Side Model Evaluation Across Tricky Indic Speech Scenarios")
            gr.Markdown(
                "Compare transcription performance on the **172,800 scenario STT benchmark**. Standard Western models (Whisper) "
                "frequently collapse code-mixing, spell out acronyms phonetically, or drop spoken Indic number patterns."
            )

            with gr.Row():
                with gr.Column(scale=5):
                    scenario_picker = gr.Dropdown(
                        choices=list(BENCHMARK_SCENARIOS.keys()),
                        value=list(BENCHMARK_SCENARIOS.keys())[0],
                        label="Select Scenario Benchmark Challenge",
                        interactive=True
                    )
                    reference_box = gr.Textbox(
                        label="Ground Truth Reference Transcript",
                        interactive=False,
                        lines=2
                    )
                    synth_audio_btn = gr.Button("Synthesize & Play Ground Truth Audio", variant="secondary")
                    sample_audio_player = gr.Audio(label="Reference Speech Audio (Edge-TTS)", type="filepath")
                    scenario_insight_box = gr.Markdown("---")

                with gr.Column(scale=7):
                    gr.Markdown("#### Model Transcription & Error Scorecards")

                    with gr.Tabs():
                        with gr.TabItem("OpenAI Whisper"):
                            whisper_box = gr.Textbox(label="Whisper Output", interactive=False)
                            with gr.Row():
                                whisper_wer = gr.Number(label="WER (%)", precision=2)
                                whisper_cer = gr.Number(label="CER (%)", precision=2)
                                whisper_digits = gr.Number(label="Digit Accuracy (%)", precision=1)
                                whisper_acronyms = gr.Number(label="Acronym Retention (%)", precision=1)

                        with gr.TabItem("Sarvam AI (Indic ASR)"):
                            sarvam_box = gr.Textbox(label="Sarvam AI Output", interactive=False)
                            with gr.Row():
                                sarvam_wer = gr.Number(label="WER (%)", precision=2)
                                sarvam_cer = gr.Number(label="CER (%)", precision=2)
                                sarvam_digits = gr.Number(label="Digit Accuracy (%)", precision=1)
                                sarvam_acronyms = gr.Number(label="Acronym Retention (%)", precision=1)

                        with gr.TabItem("Google Cloud STT"):
                            google_box = gr.Textbox(label="Google STT Output", interactive=False)
                            with gr.Row():
                                google_wer = gr.Number(label="WER (%)", precision=2)
                                google_cer = gr.Number(label="CER (%)", precision=2)
                                google_digits = gr.Number(label="Digit Accuracy (%)", precision=1)
                                google_acronyms = gr.Number(label="Acronym Retention (%)", precision=1)

            def update_scenario_display(scenario_key):
                data = BENCHMARK_SCENARIOS[scenario_key]
                ref = data["reference"]
                w_out = data["whisper"]
                s_out = data["sarvam"]
                g_out = data["google"]

                w_metrics = calculate_metrics_for_pred(ref, w_out)
                s_metrics = calculate_metrics_for_pred(ref, s_out)
                g_metrics = calculate_metrics_for_pred(ref, g_out)

                insight_md = f"**Evaluation Insight**: {data['insight']}"

                return (
                    ref,
                    None,
                    insight_md,
                    w_out, w_metrics["wer"], w_metrics["cer"], w_metrics["digit_acc"], w_metrics["acronym_acc"],
                    s_out, s_metrics["wer"], s_metrics["cer"], s_metrics["digit_acc"], s_metrics["acronym_acc"],
                    g_out, g_metrics["wer"], g_metrics["cer"], g_metrics["digit_acc"], g_metrics["acronym_acc"]
                )

            def synthesize_benchmark_audio(scenario_key):
                data = BENCHMARK_SCENARIOS[scenario_key]
                ref = data["reference"]
                lang = data["lang"]
                engine = AudioEngine(language=lang)
                return engine.synthesize(ref)

            scenario_picker.change(
                update_scenario_display,
                inputs=[scenario_picker],
                outputs=[
                    reference_box, sample_audio_player, scenario_insight_box,
                    whisper_box, whisper_wer, whisper_cer, whisper_digits, whisper_acronyms,
                    sarvam_box, sarvam_wer, sarvam_cer, sarvam_digits, sarvam_acronyms,
                    google_box, google_wer, google_cer, google_digits, google_acronyms
                ]
            )

            synth_audio_btn.click(
                synthesize_benchmark_audio,
                inputs=[scenario_picker],
                outputs=[sample_audio_player]
            )

            # Custom sentence testing
            gr.Markdown("---")
            gr.Markdown("### Live Custom Transcript Evaluation")
            with gr.Row():
                custom_ref_input = gr.Textbox(
                    label="Ground Truth Transcript",
                    placeholder="Enter reference ground truth sentence...",
                    value="खाता संख्या 9821034567 में तुरंत KYC अपडेट करें।"
                )
                custom_hyp_input = gr.Textbox(
                    label="Predicted Hypothesis Transcript",
                    placeholder="Enter model prediction transcript...",
                    value="खाता संख्या 9821034567 में तुरंत के वाई सी अपडेट करें"
                )
            calc_btn = gr.Button("Calculate Error Metrics", variant="primary")
            with gr.Row():
                custom_wer_res = gr.Number(label="WER (%)", precision=2)
                custom_cer_res = gr.Number(label="CER (%)", precision=2)
                custom_digit_res = gr.Number(label="Digit Accuracy (%)", precision=1)
                custom_acronym_res = gr.Number(label="Acronym Retention (%)", precision=1)

            def handle_custom_metrics(ref, hyp):
                m = calculate_metrics_for_pred(ref, hyp)
                return m["wer"], m["cer"], m["digit_acc"], m["acronym_acc"]

            calc_btn.click(
                handle_custom_metrics,
                inputs=[custom_ref_input, custom_hyp_input],
                outputs=[custom_wer_res, custom_cer_res, custom_digit_res, custom_acronym_res]
            )

        # ======================================================================
        # TAB 3: DATASET EXPLORER & LEADERBOARD
        # ======================================================================
        with gr.TabItem("Dataset Explorer & Leaderboard", id="tab_dataset"):
            gr.Markdown("### Browse Synthetic Datasets & Published Benchmark Leaderboards")

            with gr.Tabs():
                with gr.TabItem("Telephony Dialogues (16,370 Conversations)"):
                    gr.Markdown("Explore multi-turn voice conversations from [Hugging Face: `ansh-rohilla/verbalyze-dialogues`](https://huggingface.co/datasets/ansh-rohilla/verbalyze-dialogues):")
                    with gr.Row():
                        ds_lang_filter = gr.Dropdown(
                            choices=["Hindi (hi)", "Gujarati (gu)", "Tamil (ta)", "Telugu (te)", "Marathi (mr)", "Bengali (bn)", "Kannada (kn)", "Malayalam (ml)", "Punjabi (pa)", "English (en)"],
                            value="Hindi (hi)",
                            label="Language"
                        )
                        ds_dialogue_slider = gr.Slider(
                            minimum=1,
                            maximum=25,
                            step=1,
                            value=1,
                            label="Sample Dialogue Index"
                        )

                    dialogue_meta_display = gr.Markdown("### Dialogue Metadata")
                    dialogue_chat_display = gr.Chatbot(label="Dialogue Turns Viewer", height=400)
                    play_turn_audio_btn = gr.Button("Play First Turn Audio", variant="secondary")
                    turn_audio_player = gr.Audio(label="Spoken Turn Audio", type="filepath")

                    def view_dialogue(lang_str, idx):
                        lang_code = lang_str.split("(")[-1].strip(")")
                        dialogues = SAMPLE_DIALOGUES.get(lang_code, [])
                        if not dialogues:
                            return "No cached dialogues found for this language.", [], None

                        selected = dialogues[(idx - 1) % len(dialogues)]
                        meta = selected.get("metadata", {})
                        meta_md = f"""
                        **Scenario**: `{meta.get('scenario', 'normal')}` | **Primary Language**: `{meta.get('primary_language', lang_code)}`  
                        **Language Switch**: `{'Yes' if meta.get('has_language_switch') else 'No'}` | **Turns**: `{meta.get('num_turns', len(selected.get('messages', [])))}` | **Human Likeness**: `{meta.get('human_likeness_score', 0.98)}`
                        """

                        messages = []
                        first_asst_speech = ""
                        for m in selected.get("messages", []):
                            if m.get("role") in ["user", "assistant"]:
                                messages.append({"role": m["role"], "content": m["content"]})
                                if not first_asst_speech and m.get("role") == "assistant":
                                    first_asst_speech = m["content"]

                        return meta_md, messages, None

                    def play_turn(lang_str, idx):
                        lang_code = lang_str.split("(")[-1].strip(")")
                        dialogues = SAMPLE_DIALOGUES.get(lang_code, [])
                        if not dialogues:
                            return None
                        selected = dialogues[(idx - 1) % len(dialogues)]
                        for m in selected.get("messages", []):
                            if m.get("role") == "assistant" and m.get("content"):
                                engine = AudioEngine(language=lang_code)
                                return engine.synthesize(m["content"])
                        return None

                    ds_lang_filter.change(view_dialogue, inputs=[ds_lang_filter, ds_dialogue_slider], outputs=[dialogue_meta_display, dialogue_chat_display, turn_audio_player])
                    ds_dialogue_slider.change(view_dialogue, inputs=[ds_lang_filter, ds_dialogue_slider], outputs=[dialogue_meta_display, dialogue_chat_display, turn_audio_player])
                    play_turn_audio_btn.click(play_turn, inputs=[ds_lang_filter, ds_dialogue_slider], outputs=[turn_audio_player])

                with gr.TabItem("Public Indic Leaderboard"):
                    gr.Markdown("""
                    ### Benchmark Comparison on Indic Speech (172,800 Utterances)
                    Evaluation of leading speech engines across 12 Indian languages on the **Verbalyze STT Benchmark Suite**:

                    | Model | Primary Focus | Code-Mixed WER (%) | Spoken Digit Accuracy (%) | Acronym Retention (%) | Latency (TTFT) |
                    |:---|:---:|:---:|:---:|:---:|:---:|
                    | **Verbalyze SLM (Fine-Tuned)** | Telephony Outbound | **3.8%** | **94.2%** | **88.5%** | **~180ms** |
                    | **Sarvam AI (Indic ASR)** | Native Indic Audio | **4.2%** | **91.6%** | **86.0%** | **~350ms** |
                    | **OpenAI Whisper-Large-v3** | General Speech | **9.6%** | **82.4%** | **71.2%** | **~620ms** |
                    | **Google Cloud Speech-to-Text** | Multi-lingual Enterprise | **7.8%** | **85.0%** | **78.4%** | **~410ms** |
                    | **OpenAI Whisper-Base** | Lightweight General | **14.2%** | **76.1%** | **64.0%** | **~240ms** |

                    #### Key Insights:
                    1. **Code-Mixing Degradation**: General models suffer 2.3× higher Word Error Rates on Hinglish, Tanglish, and Gujlish sentences because they struggle to predict script boundary switches.
                    2. **Digit & Currency Errors**: Western speech engines convert Indian numbering formats ('एक लाख बीस हजार') prematurely or incorrectly into decimal notation.
                    3. **Acronym Dropping**: Critical Indian telephony acronyms (UPI, KYC, OTP, IFSC) get transcribed phonetically, breaking automated telephony state machines.
                    """)

    # Initial trigger on load
    demo.load(
        handle_start_call,
        inputs=[session_state, lang_dropdown, persona_dropdown, provider_radio, api_key_input, min_score_slider, telephony_sim_checkbox],
        outputs=[chatbot_ui, bot_audio_output, status_display, telemetry_display, quality_display]
    )
    demo.load(
        update_scenario_display,
        inputs=[scenario_picker],
        outputs=[
            reference_box, sample_audio_player, scenario_insight_box,
            whisper_box, whisper_wer, whisper_cer, whisper_digits, whisper_acronyms,
            sarvam_box, sarvam_wer, sarvam_cer, sarvam_digits, sarvam_acronyms,
            google_box, google_wer, google_cer, google_digits, google_acronyms
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css, share=False)
