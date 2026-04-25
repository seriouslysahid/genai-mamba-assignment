"""
Sanity check script to verify the Python 3.10 + PyTorch + Mamba environment.

Usage:
    python scripts/check_env.py
"""

import sys

def main():
    print("--- Environment Check ---")
    
    # 1. Python version
    py_version = sys.version_info
    print(f"Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major != 3 or py_version.minor < 10:
        print(f"  [WARNING] Python 3.10+ required, got {py_version.major}.{py_version.minor}")
    else:
        print(f"  [OK] Python {py_version.major}.{py_version.minor}")

    # 2. CUDA & BF16
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if not torch.cuda.is_available():
            print("  [ERROR] CUDA is not available. Check your PyTorch installation or GPU drivers.")
            sys.exit(1)
        
        print(f"CUDA Available: Yes ({torch.version.cuda})")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        capability = torch.cuda.get_device_capability(0)
        print(f"GPU capability: sm_{capability[0]}{capability[1]}")
        
        if capability[0] < 8:
            print("  [INFO] Use float16 on this GPU. BF16 kernels require sm_80+.")
        elif not torch.cuda.is_bf16_supported():
            print("  [WARNING] GPU does not support bfloat16. Training might be slower.")
        else:
            print("  [OK] bfloat16 is supported.")
            
    except ImportError:
        print("  [ERROR] PyTorch is not installed.")
        sys.exit(1)

    # 3. Mamba dependencies
    try:
        import causal_conv1d
        print(f"causal_conv1d: {causal_conv1d.__version__}")
        print("  [OK] causal_conv1d imported successfully.")
    except ImportError as e:
        print(f"  [ERROR] Failed to import causal_conv1d: {e}")
    
    try:
        import mamba_ssm
        print(f"mamba_ssm: {mamba_ssm.__version__}")
        print("  [OK] mamba_ssm imported successfully.")
    except ImportError as e:
        print(f"  [ERROR] Failed to import mamba_ssm: {e}")
    
    try:
        import torch
        from mamba_ssm import Mamba
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        m = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).cuda().to(dtype)
        x = torch.randn(1, 32, 128, device="cuda", dtype=dtype)
        y = m(x)
        assert y.shape == x.shape
        del m, x, y
        torch.cuda.empty_cache()
        print("  [OK] Mamba CUDA kernel forward pass.")
    except Exception as e:
        print(f"  [ERROR] Mamba kernel smoke test failed: {e}")
        sys.exit(1)

    print("--- Check Complete ---")

if __name__ == "__main__":
    main()
