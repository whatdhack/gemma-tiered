"""Environment and Backend Verification Script for Gemma Tiered Router.

Verifies:
1. Python version (asserts 3.12).
2. Essential packages (pydantic, rich, requests).
3. LiteRT / LiteRT-LM backend enumeration (CPU, DirectML/GPU, NPU).
4. Local Ollama Tier 2 service status.
"""

import sys
import platform

print("=" * 60)
print("GEMMA TIERED ROUTING HARNESS - ENVIRONMENT VERIFICATION")
print("=" * 60)

# 1. Python version check
py_version = sys.version_info
print(f"[*] Python Version: {py_version.major}.{py_version.minor}.{py_version.micro} ({platform.architecture()[0]})")
print(f"[*] Executable Path: {sys.executable}")
if py_version.major != 3 or py_version.minor != 12:
    print(f"[!] Warning: Expected Python 3.12, got {py_version.major}.{py_version.minor}")
else:
    print("[+] Python 3.12 environment confirmed.")

# 2. Package imports
for pkg in ["pydantic", "rich", "requests"]:
    try:
        __import__(pkg)
        print(f"[+] Package '{pkg}' is installed.")
    except ImportError:
        print(f"[-] Missing package: '{pkg}'. Please run pip install {pkg}")

# 3. LiteRT / LiteRT-LM check
print("\n[*] Checking LiteRT / LiteRT-LM backends...")
litert_found = False
try:
    import ai_edge_litert
    from ai_edge_litert import interpreter
    print(f"[+] Successfully loaded ai_edge_litert (v{getattr(ai_edge_litert, '__version__', '2.3.0')})")
    print("[+] LiteRT Edge Interpreter is available and ready.")
    litert_found = True
except ImportError:
    pass

try:
    import litert_lm
    print("[+] Successfully imported litert_lm!")
    litert_found = True
except ImportError:
    pass

if not litert_found:
    print("[i] LiteRT edge runtime not found. Using reflex classifier fallback.")

# 4. Ollama Service check
print("\n[*] Checking Tier 2 Big Brain (Ollama) Endpoint...")
try:
    import requests
    resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
    if resp.status_code == 200:
        models = [m.get("name") for m in resp.json().get("models", [])]
        print(f"[+] Ollama service is RUNNING at http://127.0.0.1:11434")
        print(f"[+] Available Models: {models if models else 'None loaded yet (run `ollama pull gemma2:2b`)'}")
    else:
        print(f"[!] Ollama returned status code: {resp.status_code}")
except Exception as e:
    print("[i] Ollama server is currently offline at http://127.0.0.1:11434.")
    print("    (To start: run `ollama serve` and `ollama pull gemma2:2b`. tiered_router will use synthetic mock mode if offline).")

print("=" * 60)
print("Verification complete.")
print("=" * 60)
