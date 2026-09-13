"""
verbalyze/cli.py

Master command-line interface for the Verbalyze Indic Voice AI Suite:
  verbalyze export-stt          # Compile 172.8k STT benchmark dataset & Hugging Face card
  verbalyze export-dialogues    # Compile 16.3k multi-turn voice conversations (ChatML/ShareGPT)
  verbalyze benchmark           # Run ASR benchmark evaluation across 11 scenarios
  verbalyze agent               # Start interactive telephony voice agent simulator
  verbalyze server              # Run FastAPI telephony webhook server for SIP trunks
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="verbalyze",
        description="Verbalyze: Indic Voice AI Suite for Datasets, SLM Training & Telephony Bots"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: export-stt
    p_stt = subparsers.add_parser("export-stt", help="Package 172,800 STT dataset for Hugging Face")
    p_stt.add_argument("--stt-dir", type=str, default="stt_dataset", help="Path to STT dataset directory")
    p_stt.add_argument("--output-dir", type=str, default="data/stt_bench", help="Output directory")
    p_stt.add_argument("--langs", type=str, default=None, help="Comma-separated language codes (e.g. hi,en,ta)")

    # Command: export-dialogues
    p_dlg = subparsers.add_parser("export-dialogues", help="Package 16,370 conversations for SLM fine-tuning")
    p_dlg.add_argument("--repo-root", type=str, default=".", help="Root directory containing dataset_*.json")
    p_dlg.add_argument("--output-dir", type=str, default="data/dialogues", help="Output directory")

    # Command: benchmark
    p_bm = subparsers.add_parser("benchmark", help="Run ASR evaluation benchmark")
    p_bm.add_argument("--lang", type=str, default="hi", help="Language code to evaluate (default: hi)")
    p_bm.add_argument("--samples", type=int, default=20, help="Samples per scenario to evaluate")
    p_bm.add_argument("--model-name", type=str, default="Mock-Whisper-Base", help="Model name for leaderboard")
    p_bm.add_argument("--output-report", type=str, default="LEADERBOARD.md", help="Markdown leaderboard path")

    # Command: agent
    p_ag = subparsers.add_parser("agent", help="Launch interactive telephony voicebot simulator")
    p_ag.add_argument("--lang", type=str, default="hi", choices=["hi", "en", "ta", "te", "mr", "gu", "bn"], help="Language code")
    p_ag.add_argument("--no-voice", action="store_true", help="Disable audio speech synthesis playback")
    p_ag.add_argument("--provider", type=str, default="groq", choices=["groq", "openai", "mock"], help="LLM Provider")

    # Command: server
    p_srv = subparsers.add_parser("server", help="Start FastAPI telephony webhook server")
    p_srv.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    p_srv.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")

    # Command: publish-hf
    p_hf = subparsers.add_parser("publish-hf", help="Publish datasets to Hugging Face Hub")
    p_hf.add_argument("--target", type=str, default="all", choices=["all", "dialogues", "stt"],
                      help="Dataset to publish: 'dialogues', 'stt', or 'all' (default: all)")
    p_hf.add_argument("--token", type=str, default=None, help="Hugging Face API token (with Write permissions)")
    p_hf.add_argument("--private", action="store_true", help="Publish as private dataset")
    p_hf.add_argument("--repo-id", type=str, default=None, help="Custom repo ID (e.g. username/my-dataset)")

    # Command: ui
    p_ui = subparsers.add_parser("ui", help="Launch interactive Gradio web application")
    p_ui.add_argument("--port", type=int, default=7860, help="Web app port (default: 7860)")
    p_ui.add_argument("--share", action="store_true", help="Generate a public Gradio share link")

    # Command: deploy-space
    p_sp = subparsers.add_parser("deploy-space", help="Deploy web application to Hugging Face Spaces")
    p_sp.add_argument("--repo-id", type=str, default="ansh-rohilla/verbalyze-demo", help="Target Space repo ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "export-stt":
        from verbalyze.pipeline.stt_exporter import export_stt_dataset
        langs = [l.strip() for l in args.langs.split(",")] if args.langs else None
        export_stt_dataset(stt_base_dir=args.stt_dir, output_dir=args.output_dir, languages=langs)

    elif args.command == "export-dialogues":
        from verbalyze.pipeline.dialogue_exporter import export_dialogue_dataset
        export_dialogue_dataset(repo_root=args.repo_root, output_dir=args.output_dir)

    elif args.command == "benchmark":
        from verbalyze.benchmark.evaluator import BenchmarkRunner, MockASRClient
        runner = BenchmarkRunner()
        client = MockASRClient(accuracy_level=0.90)
        res = runner.run_benchmark(
            model_name=args.model_name,
            client=client,
            lang=args.lang,
            limit_per_scenario=args.samples
        )
        runner.generate_leaderboard_markdown([res], output_file=args.output_report)

    elif args.command == "agent":
        from verbalyze.agent.voice_bot import VoiceAgent
        bot = VoiceAgent(language=args.lang, llm_provider=args.provider, voice_enabled=not args.no_voice)
        bot.run_interactive_terminal_call()

    elif args.command == "server":
        try:
            import uvicorn
            from verbalyze.telephony.server import create_app
            app = create_app()
            print(f"[Server] Starting Verbalyze Telephony Webhook Server on {args.host}:{args.port}...")
            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError as e:
            print(f"Error starting server: {e}")
            print("Install uvicorn and fastapi: pip install uvicorn fastapi")

    elif args.command == "publish-hf":
        from verbalyze.pipeline.hf_publisher import publish_dataset, publish_all
        try:
            if args.target == "dialogues":
                publish_dataset(
                    folder_path="data/dialogues",
                    repo_name="verbalyze-dialogues",
                    token=args.token,
                    private=args.private,
                    custom_repo_id=args.repo_id
                )
            elif args.target == "stt":
                publish_dataset(
                    folder_path="data/stt_bench",
                    repo_name="verbalyze-stt-bench",
                    token=args.token,
                    private=args.private,
                    custom_repo_id=args.repo_id
                )
            else:
                publish_all(token=args.token, private=args.private)
        except Exception as e:
            print(f"\n❌ Error publishing to Hugging Face: {e}")
            sys.exit(1)

    elif args.command == "ui":
        try:
            import app
            print(f"[UI] Starting Verbalyze Interactive Web App on http://localhost:{args.port}...")
            app.demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
        except Exception as e:
            print(f"❌ Error launching web application: {e}")
            sys.exit(1)

    elif args.command == "deploy-space":
        from scripts.deploy_space import deploy
        deploy(repo_id=args.repo_id)


if __name__ == "__main__":
    main()
