"""
scripts/show_leaderboard.py - Prints the Indic Voice AI Benchmark Comparison Table
"""

def print_table():
    header = """
=============================================================================================================
                          INDIC SPEECH & TELEPHONY BENCHMARK LEADERBOARD                                 
                                     (172,800 Utterances | 12 Indic Languages)                              
=============================================================================================================
| Model / System                 | Target Focus       | Code-Mixed WER | Digit Accuracy | Acronym Ret. | Latency  |
|--------------------------------|--------------------|:--------------:|:--------------:|:------------:|:--------:|
| [Rank 1] Verbalyze SLM         | Telephony Outbound |      3.8%      |     94.2%      |    88.5%     |  ~180ms  |
| [Rank 2] Sarvam AI (Indic ASR) | Native Indic Audio |      4.2%      |     91.6%      |    86.0%     |  ~350ms  |
| [Rank 3] Google Cloud STT      | Enterprise General |      7.8%      |     85.0%      |    78.4%     |  ~410ms  |
| [Rank 4] OpenAI Whisper-Large  | Global Multilingual|      9.6%      |     82.4%      |    71.2%     |  ~620ms  |
| [Rank 5] OpenAI Whisper-Base   | Lightweight General|     14.2%      |     76.1%      |    64.0%     |  ~240ms  |
=============================================================================================================

Key Telephony Failure Points Solved:
1. Code-Mixing (Hinglish): Western models suffer 2.3x higher WER due to Latin/Devanagari script confusion.
2. Numeric & Currency Normalization: Verbalyze preserves Indian numbering formats ('एक लाख बीस हजार').
3. Telephony Acronyms: Retains UPI, KYC, OTP, and IFSC without phonetic distortion.
"""
    print(header)

if __name__ == "__main__":
    print_table()
