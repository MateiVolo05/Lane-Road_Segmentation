import os
import re
import sys
import shutil
import platform
import subprocess


def run_command(cmd, description):
    print(f"\n[INFO] {description}...")
    print(f"       Command: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def detect_nvidia_gpu():
    """
    Detects if an NVIDIA GPU and drivers are available using nvidia-smi.
    Returns (has_gpu: bool, cuda_version: float or None, gpu_name: str or None).
    """
    if shutil.which("nvidia-smi") is not None:
        try:
            output = subprocess.check_output(
                ["nvidia-smi"], stderr=subprocess.STDOUT, text=True
            )
            cuda_match = re.search(r"CUDA Version:\s*(\d+\.\d+)", output)
            cuda_ver = float(cuda_match.group(1)) if cuda_match else None

            gpu_name = None
            try:
                name_out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    stderr=subprocess.STDOUT,
                    text=True,
                ).strip()
                if name_out:
                    gpu_name = name_out.splitlines()[0]
            except Exception:
                pass

            return True, cuda_ver, gpu_name
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Fallback check: see if an NVIDIA GPU exists in hardware without drivers installed
    system = platform.system()
    try:
        if system == "Windows":
            ps_cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ]
            out = subprocess.check_output(ps_cmd, stderr=subprocess.DEVNULL, text=True)
            if "NVIDIA" in out.upper():
                print(
                    "[WARNING] NVIDIA GPU detected, but no 'nvidia-smi' was found"
                )
                print(
                    "          Install official NVIDIA drivers before running"
                )
        elif system == "Linux" and shutil.which("lspci"):
            out = subprocess.check_output(["lspci"], stderr=subprocess.DEVNULL, text=True)
            if "NVIDIA" in out.upper():
                print(
                    "[WARNING] NVIDIA GPU detected (lspci), but driver 'nvidia-smi' is not active."
                )
    except Exception:
        pass

    return False, None, None


def get_pytorch_install_cmd(has_gpu, cuda_ver):
    """
    Builds the pip install command for torch and torchvision based on detected hardware.
    """
    base_cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision"]
    system = platform.system()

    if has_gpu:
        # Select appropriate CUDA wheel repository based on driver CUDA support
        if cuda_ver is None or cuda_ver >= 12.4:
            index_url = "https://download.pytorch.org/whl/cu124"
        elif cuda_ver >= 12.1:
            index_url = "https://download.pytorch.org/whl/cu121"
        else:
            index_url = "https://download.pytorch.org/whl/cu118"

        print(f"[INFO] Configuring PyTorch with GPU (CUDA) support from: {index_url}")
        return base_cmd + ["--index-url", index_url]
    else:
        if system in ("Linux", "Windows"):
            # Explicit CPU index avoids downloading ~2.5 GB of unused CUDA packages on Linux
            index_url = "https://download.pytorch.org/whl/cpu"
            print(f"[INFO] No GPU NVIDIA detected. Optimised CPU variant is being downloaded: {index_url}")
            return base_cmd + ["--index-url", index_url]
        else:
            print("[INFO] Standard PyTorch is downloading.")
            return base_cmd


def verify_installation():
    """
    Imports installed packages and prints a final hardware & environment summary.
    """
    print("\n" + "=" * 55)
    print("             CHECK INSTALLED ENVIRONMENT")
    print("=" * 55)
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np

        print(f"Python version      : {platform.python_version()}")
        print(f"PyTorch version     : {torch.__version__}")
        print(f"Torchvision version : {torchvision.__version__}")
        print(f"OpenCV version      : {cv2.__version__}")
        print(f"NumPy version       : {np.__version__}")
        print(f"CUDA available      : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Active GPU          : {torch.cuda.get_device_name(0)}")
        else:
            print("Active Device       : CPU")
    except Exception as e:
        print(f"[ERROR] Packet verification error: {e}")
    print("=" * 55 + "\n")


def main():
    print("=" * 55)
    print("   AUTOMATED DEPENDENCIES CONFIGURATION - LANE DETECTION")
    print("=" * 55)
    print(f"OS : {platform.system()} {platform.release()}")
    print(f"Python Environment    : {sys.executable}")

    # 1. Upgrade pip
    run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Updating pip to latest version",
    )

    # 2. Detect hardware (NVIDIA GPU vs CPU)
    has_gpu, cuda_ver, gpu_name = detect_nvidia_gpu()
    if has_gpu:
        print(f"\n[HARDWARE] GPU detected: {gpu_name or 'NVIDIA GPU'} (supported CUDA: {cuda_ver or 'N/A'})")
    else:
        print("\n[HARDWARE] CPU configuration will be made")

    # 3. Install PyTorch & Torchvision matching the hardware
    pytorch_cmd = get_pytorch_install_cmd(has_gpu, cuda_ver)
    run_command(pytorch_cmd, "Installing PyTorch and Torchvision")

    # 4. Install remaining project dependencies (main.py + LaneATT + TuSimple metrics)
    other_packages = [
        "opencv-python",
        "numpy",
        "tqdm",
        "scipy",         # required by LaneATT/lib/lane.py (InterpolatedUnivariateSpline)
        "scikit-learn",  # required by LaneATT/utils/tusimple_metric.py (LinearRegression)
        "ujson",         # required by LaneATT/utils/tusimple_metric.py
        "pyyaml",        # required by LaneATT/lib/config.py
    ]
    run_command(
        [sys.executable, "-m", "pip", "install"] + other_packages,
        "Project dependency installations (OpenCV, NumPy, SciPy, Scikit-Learn, tqdm, etc.)",
    )

    # 5. Final verification
    verify_installation()


if __name__ == "__main__":
    main()