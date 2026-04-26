import os
import time
import asyncio
import httpx
import subprocess
from huggingface_hub import HfApi

# Configurations
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("HF_TOKEN is missing!")
    exit(1)

SPACE_ID = "SmartKapila/OmniGuard-Evolved-V2"
LOCAL_DIR = "/home/harsheen/metahack/OmniGuard-Evolved-V2"

print(f"Deploying {LOCAL_DIR} to HF Space {SPACE_ID}...")

# Start logging in the background using curl
def start_log_stream(log_type):
    url = f"https://huggingface.co/api/spaces/{SPACE_ID}/logs/{log_type}"
    cmd = ["curl", "-N", "-H", f"Authorization: Bearer {HF_TOKEN}", url]
    print(f"Starting {log_type} log stream: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

build_logs = start_log_stream("build")
run_logs = start_log_stream("run")

api = HfApi(token=HF_TOKEN)
try:
    api.upload_folder(
        folder_path=LOCAL_DIR,
        path_in_repo=".",
        repo_id=SPACE_ID,
        repo_type="space",
        ignore_patterns=[".git", "__pycache__", ".ipynb_checkpoints"]
    )
    print("Code successfully pushed to Hugging Face Space.")
except Exception as e:
    print(f"Deployment failed: {e}")

print("Waiting a bit for Space to potentially restart and load...")
time.sleep(30) # Wait to allow building/running

# Stress test
async def stress_test():
    url = f"https://{SPACE_ID.replace('/', '-').lower()}.hf.space/api/infer"
    print(f"Starting stress test against {url}")
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "payload": "Stress test payload malicious input command injection.",
        "model_type": "base"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []
        for i in range(20):
            tasks.append(client.post(url, json=payload, headers=headers))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success = 0
        failed = 0
        for r in results:
            if isinstance(r, Exception):
                failed += 1
                print(f"Failure: {r}")
            elif r.status_code == 200:
                success += 1
            else:
                failed += 1
                print(f"Failure status: {r.status_code} - {r.text}")
        print(f"Stress test completed: {success} successes, {failed} failures.")

asyncio.run(stress_test())

print("Reading final log outputs...")
# Let the logs buffer a bit
time.sleep(5)
if build_logs.poll() is None:
    build_logs.terminate()
if run_logs.poll() is None:
    run_logs.terminate()

out_build, _ = build_logs.communicate()
out_run, _ = run_logs.communicate()

print("\n--- BUILD LOGS ---")
if out_build:
    print(out_build[-2000:])  # last 2000 chars

print("\n--- RUN LOGS ---")
if out_run:
    print(out_run[-2000:])  # last 2000 chars

print("Deployment and testing complete!")
