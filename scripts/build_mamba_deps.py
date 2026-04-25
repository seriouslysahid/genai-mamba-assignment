"""
Build and install causal-conv1d and mamba-ssm from source for A100 (sm_80).

Usage:
    python scripts/build_mamba_deps.py
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Set environment variables before any imports
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ["CAUSAL_CONV1D_FORCE_BUILD"] = "1"
os.environ["MAMBA_FORCE_BUILD"] = "1"
os.environ["MAX_JOBS"] = "8"

import torch


def check_prerequisites():
    """Verify nvcc, git, CUDA availability, and A100 compute capability."""
    print("--- Checking prerequisites ---")
    
    # Check nvcc
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        print(f"✓ nvcc found: {result.stdout.splitlines()[-1].strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ ERROR: nvcc not found. Install CUDA toolkit.")
        sys.exit(1)
    
    # Check git
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True, check=True)
        print(f"✓ git found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ ERROR: git not found. Install git.")
        sys.exit(1)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("✗ ERROR: torch.cuda.is_available() returned False.")
        sys.exit(1)
    print(f"✓ CUDA available: {torch.version.cuda}")
    
    # Check compute capability
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 0):
        print(f"✗ ERROR: Expected compute capability (8, 0) for A100, got {capability}")
        sys.exit(1)
    print(f"✓ GPU compute capability: sm_{capability[0]}{capability[1]} (A100)")
    
    print()


def clone_or_update_repo(repo_url, target_dir):
    """Clone repo with --depth 1 or pull if it already exists."""
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = target_dir / repo_name
    
    if repo_path.exists():
        print(f"Repository {repo_name} already exists, pulling latest...")
        subprocess.run(["git", "-C", str(repo_path), "pull"], check=True)
    else:
        print(f"Cloning {repo_name}...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_path)], check=True)
    
    return repo_path


def patch_setup_py(setup_path):
    """Patch setup.py to set TORCH_CUDA_ARCH_LIST to 8.0."""
    print(f"Patching {setup_path}...")
    
    with open(setup_path, "r") as f:
        content = f.read()
    
    # Check if TORCH_CUDA_ARCH_LIST assignment already exists
    existing_pattern = r'os\.environ\["TORCH_CUDA_ARCH_LIST"\]\s*=\s*["\'][^"\']*["\']'
    
    if re.search(existing_pattern, content):
        # Replace existing assignment
        content = re.sub(existing_pattern, 'os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"', content)
        print("  → Replaced existing TORCH_CUDA_ARCH_LIST assignment")
    else:
        # Prepend after first "import os"
        import_os_pattern = r'(import\s+os\s*\n)'
        match = re.search(import_os_pattern, content)
        
        if match:
            insert_pos = match.end()
            content = (content[:insert_pos] + 
                      'os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"\n' + 
                      content[insert_pos:])
            print("  → Added TORCH_CUDA_ARCH_LIST assignment after 'import os'")
        else:
            print("  ⚠ WARNING: Could not find 'import os' in setup.py")
    
    with open(setup_path, "w") as f:
        f.write(content)


def install_package(repo_path):
    """Install package with pip using --no-build-isolation --no-deps."""
    print(f"Installing {repo_path.name}...")
    
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = "8.0"
    env["MAX_JOBS"] = "8"
    
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(repo_path)],
        env=env,
        check=True
    )
    print(f"✓ {repo_path.name} installed successfully\n")


def smoke_test():
    """Run a smoke test: import Mamba, construct model, run forward pass."""
    print("--- Running smoke test ---")
    
    try:
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        print("✓ Mamba imports successful")
        
        # Small test config
        config = MambaConfig(
            d_model=256,
            n_layer=2,
            vocab_size=1000,
            ssm_cfg={"d_state": 16}
        )
        model = MambaLMHeadModel(config).cuda()
        print("✓ Mamba model constructed on CUDA")
        
        # Forward pass
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device="cuda")
        
        with torch.no_grad():
            output = model(input_ids)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, 1000)
        assert output.logits.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {output.logits.shape}"
        
        print(f"✓ Forward pass successful: input {input_ids.shape} → output {output.logits.shape}")
        print("✓ Smoke test passed!\n")
        
    except Exception as e:
        print(f"✗ ERROR: Smoke test failed: {e}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("Building Mamba dependencies from source for A100 (sm_80)")
    print("=" * 60)
    print()
    
    check_prerequisites()
    
    # Create temp directory for cloning
    work_dir = Path.cwd()
    temp_dir = work_dir / "temp_mamba_build"
    temp_dir.mkdir(exist_ok=True)
    print(f"Working directory: {temp_dir}\n")
    
    # Clone/update repositories
    repos = [
        ("https://github.com/Dao-AILab/causal-conv1d.git", "causal-conv1d"),
        ("https://github.com/state-spaces/mamba.git", "mamba"),
    ]
    
    repo_paths = []
    for repo_url, _ in repos:
        repo_path = clone_or_update_repo(repo_url, temp_dir)
        repo_paths.append(repo_path)
        print()
    
    # Patch setup.py files
    for repo_path in repo_paths:
        setup_path = repo_path / "setup.py"
        if setup_path.exists():
            patch_setup_py(setup_path)
        else:
            print(f"⚠ WARNING: {setup_path} not found")
        print()
    
    # Install packages
    for repo_path in repo_paths:
        install_package(repo_path)
    
    # Run smoke test
    smoke_test()
    
    print("=" * 60)
    print("✓ All dependencies built and installed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
