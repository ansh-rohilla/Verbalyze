"""
scripts/generate_tts_samples.py

Generates and compares audio samples across all available Text-to-Speech (TTS) contenders:
1. Edge-TTS (hi-IN-SwaraNeural) - Female Hindi Neural
2. Edge-TTS (hi-IN-MadhurNeural) - Male Hindi Neural
3. Edge-TTS (en-IN-NeerjaNeural) - Indian English Neural
4. Google TTS (gTTS Hindi)
5. Google TTS (gTTS Indian English)
6. macOS Native Speech (Lekha hi_IN)
7. macOS Native Speech (Rishi en_IN)
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "samples" / "tts"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

HINDI_PROMPT = "नमस्कार, मैं मुथूट फिनकॉर्प से बोल रही हूँ। क्या आप आज अपनी बकाया ईएमआई ₹5,420 जमा कर पाएंगे?"
HINGLISH_PROMPT = "हाँ जी, मैंने आपके registered mobile number पर UPI payment link भेज दिया है, please check कर लीजिए।"
ENGLISH_PROMPT = "Hello, I am calling from Muthoot Fincorp regarding your overdue EMI payment of 5,420 rupees."


async def generate_edge_tts(text: str, voice: str, out_file: Path):
    import edge_tts
    comm = edge_tts.Communicate(text, voice)
    await comm.save(str(out_file))


def generate_gtts(text: str, lang: str, out_file: Path, tld: str = "co.in"):
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang, tld=tld)
    tts.save(str(out_file))


def generate_mac_say(text: str, voice: str, out_file: Path):
    aiff_path = out_file.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, text, "-o", str(aiff_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # convert to mp3 or wav using ffmpeg if available
    subprocess.run(["ffmpeg", "-y", "-i", str(aiff_path), str(out_file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if aiff_path.exists():
        aiff_path.unlink()


async def main():
    print(f"🔊 Generating TTS Samples in: {SAMPLE_DIR}\n")
    results = []

    # 1. Edge-TTS Swara (Hindi Female)
    f1 = SAMPLE_DIR / "1_edgetts_swara_hindi.mp3"
    print("Generating 1. Edge-TTS Swara (Hindi Female)...")
    await generate_edge_tts(HINDI_PROMPT, "hi-IN-SwaraNeural", f1)
    results.append(("Microsoft Edge-TTS", "hi-IN-SwaraNeural", "Hindi", "Neural / Conversational", f1))

    # 2. Edge-TTS Madhur (Hindi Male)
    f2 = SAMPLE_DIR / "2_edgetts_madhur_hindi.mp3"
    print("Generating 2. Edge-TTS Madhur (Hindi Male)...")
    await generate_edge_tts(HINDI_PROMPT, "hi-IN-MadhurNeural", f2)
    results.append(("Microsoft Edge-TTS", "hi-IN-MadhurNeural", "Hindi", "Neural / Professional", f2))

    # 3. Edge-TTS Neerja (Hinglish/Indian English)
    f3 = SAMPLE_DIR / "3_edgetts_neerja_hinglish.mp3"
    print("Generating 3. Edge-TTS Neerja (Hinglish)...")
    await generate_edge_tts(HINGLISH_PROMPT, "en-IN-NeerjaNeural", f3)
    results.append(("Microsoft Edge-TTS", "en-IN-NeerjaNeural", "Hinglish / En-IN", "Neural / Natural Code-Switch", f3))

    # 4. Google TTS (gTTS Hindi)
    f4 = SAMPLE_DIR / "4_google_gtts_hindi.mp3"
    print("Generating 4. Google TTS (gTTS Hindi)...")
    generate_gtts(HINDI_PROMPT, "hi", f4)
    results.append(("Google Translate TTS", "Standard hi", "Hindi", "Concatenative / Legacy", f4))

    # 5. Google TTS (gTTS English - India)
    f5 = SAMPLE_DIR / "5_google_gtts_english.mp3"
    print("Generating 5. Google TTS (gTTS English India)...")
    generate_gtts(ENGLISH_PROMPT, "en", f5, tld="co.in")
    results.append(("Google Translate TTS", "Standard en (co.in)", "Indian English", "Concatenative / Standard", f5))

    # 6. macOS Native Say (Lekha Hindi)
    f6 = SAMPLE_DIR / "6_macos_lekha_hindi.mp3"
    print("Generating 6. macOS Say Lekha (Hindi)...")
    generate_mac_say(HINDI_PROMPT, "Lekha", f6)
    results.append(("Apple macOS Speech", "Lekha", "Hindi", "System Native Offline", f6))

    # 7. macOS Native Say (Rishi Indian English)
    f7 = SAMPLE_DIR / "7_macos_rishi_english.mp3"
    print("Generating 7. macOS Say Rishi (Indian English)...")
    generate_mac_say(ENGLISH_PROMPT, "Rishi", f7)
    results.append(("Apple macOS Speech", "Rishi", "Indian English", "System Native Offline", f7))

    print("\n" + "=" * 80)
    print("SAMPLE GENERATION SUMMARY:")
    print("=" * 80)
    for engine, voice, lang, style, p in results:
        size_kb = round(os.path.getsize(p) / 1024, 1) if p.exists() else 0
        print(f"• [{engine}] {voice} ({lang}) | {style} -> {p.name} ({size_kb} KB)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
