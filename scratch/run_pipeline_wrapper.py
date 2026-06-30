import sys
import subprocess
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Wrapper to run generate_v4_tts_batched.py sequentially in fresh processes to prevent memory leak")
    parser.add_argument("--languages", required=True, help="Comma-separated language codes to process")
    parser.add_argument("--steps", type=int, default=10, help="Number of diffusion steps for synthesis")
    parser.add_argument("--max-records", type=int, default=1000, help="Number of records to synthesize per subprocess run")
    parser.add_argument("--sample", type=int, default=0, help="Sample limit")
    parser.add_argument("--local-only", action="store_true", help="Synthesize and save audio files locally without uploading to S3")
    parser.add_argument("--device", default="auto", help="Device to run on")
    parser.add_argument("--range-start", type=int, default=0)
    parser.add_argument("--range-end", type=int, default=0)
    parser.add_argument("--output-suffix", default="")
    
    args = parser.parse_args()
    
    cmd = [
        "python3", "-u", "scratch/generate_v4_tts_batched.py",
        "--languages", args.languages,
        "--steps", str(args.steps),
        "--max-records", str(args.max_records),
        "--batch-size", "4",
        "--device", args.device,
        "--range-start", str(args.range_start),
        "--range-end", str(args.range_end),
        "--output-suffix", args.output_suffix
    ]
    if args.sample > 0:
        cmd.extend(["--sample", str(args.sample)])
    if args.local_only:
        cmd.append("--local-only")
        
    print(f"Starting wrapper script. Running command: {' '.join(cmd)}")
    
    run_idx = 1
    while True:
        print(f"\n==============================================")
        print(f"Wrapper Run #{run_idx} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"==============================================")
        
        import os
        env = os.environ.copy()
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        # Set thread limits for safety
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        env["OPENBLAS_NUM_THREADS"] = "4"
        env["VECLIB_MAXIMUM_THREADS"] = "4"
        env["NUMEXPR_NUM_THREADS"] = "4"
        
        process = subprocess.Popen(cmd, env=env)
        exit_code = process.wait()
        
        print(f"\nSubprocess run #{run_idx} exited with code: {exit_code}")
        
        if exit_code == 10:
            print("Subprocess hit max-records limit. Starting next process to clear memory...")
            run_idx += 1
            time.sleep(2)
        elif exit_code == 0:
            print("Subprocess completed all languages successfully! Exiting wrapper.")
            break
        else:
            print(f"Subprocess failed with unexpected exit code {exit_code}. Terminating wrapper.")
            sys.exit(exit_code)

if __name__ == "__main__":
    main()
