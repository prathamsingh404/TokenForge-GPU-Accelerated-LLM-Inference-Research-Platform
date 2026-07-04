# TokenForge GPU-Accelerated LLM Inference Platform
"""
Environment validation for TokenForge.

Checks that all required dependencies and hardware are available
before running any experiments. Run this first after cloning.
"""

import sys
import importlib
import subprocess


REQUIRED_PACKAGES = {
    "torch": "PyTorch (GPU compute)",
    "transformers": "HuggingFace Transformers (model loading)",
    "accelerate": "Accelerate (device placement)",
    "bitsandbytes": "BitsAndBytes (quantization)",
    "pynvml": "PyNVML (GPU monitoring)",
    "fastapi": "FastAPI (dashboard backend)",
    "uvicorn": "Uvicorn (ASGI server)",
    "aiosqlite": "aiosqlite (async database)",
    "matplotlib": "Matplotlib (plotting)",
    "numpy": "NumPy (numerics)",
    "pandas": "Pandas (data analysis)",
    "rich": "Rich (terminal formatting)",
    "tqdm": "tqdm (progress bars)",
    "psutil": "psutil (system metrics)",
}

OPTIONAL_PACKAGES = {
    "triton": "Triton (GPU kernel compiler — Linux/WSL only)",
}


def check_python_version():
    v = sys.version_info
    print(f"  Python {v.major}.{v.minor}.{v.micro}", end="")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print(" — FAIL (need >=3.10)")
        return False
    print(" — OK")
    return True


def check_package(name, description):
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {description}: {version} — OK")
        return True
    except ImportError:
        print(f"  {description}: NOT INSTALLED")
        return False


def check_cuda():
    try:
        import torch
        if not torch.cuda.is_available():
            print("  CUDA: Not available")
            return False

        dev = torch.cuda.get_device_properties(0)
        print(f"  GPU: {dev.name}")
        print(f"  VRAM: {dev.total_mem // (1024**2)} MB")
        print(f"  Compute Capability: {dev.major}.{dev.minor}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  SM Count: {dev.multi_processor_count}")
        return True
    except Exception as e:
        print(f"  CUDA check failed: {e}")
        return False


def check_node():
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        version = result.stdout.strip()
        print(f"  Node.js: {version} — OK")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  Node.js: NOT INSTALLED (needed for dashboard frontend)")
        return False


def main():
    print("=" * 60)
    print("TokenForge — Environment Check")
    print("=" * 60)

    all_ok = True

    print("\n[1/5] Python Version")
    all_ok &= check_python_version()

    print("\n[2/5] CUDA / GPU")
    all_ok &= check_cuda()

    print("\n[3/5] Required Packages")
    missing = []
    for pkg, desc in REQUIRED_PACKAGES.items():
        if not check_package(pkg, desc):
            missing.append(pkg)
            all_ok = False

    print("\n[4/5] Optional Packages")
    for pkg, desc in OPTIONAL_PACKAGES.items():
        check_package(pkg, desc)

    print("\n[5/5] Node.js (for dashboard)")
    check_node()

    print("\n" + "=" * 60)
    if all_ok:
        print("All checks passed. Ready to run experiments.")
    else:
        print("Some checks failed.")
        if missing:
            pkg_list = " ".join(missing)
            print(f"\nInstall missing packages:\n  pip install {pkg_list}")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
