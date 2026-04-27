# build mamba deps from source
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ["CAUSAL_CONV1D_FORCE_BUILD"] = "1"
os.environ["MAMBA_FORCE_BUILD"] = "1"
os.environ["MAX_JOBS"] = "8"

import torch

def check_prereqs():
    print("checking nvcc...")
    try:
        subprocess.run(["nvcc", "--version"], capture_output=True, check=True)
    except:
        print("nvcc not found")
        sys.exit(1)
    
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except:
        print("git not found")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("cuda not available")
        sys.exit(1)
        
    cap = torch.cuda.get_device_capability(0)
    if cap != (8, 0):
        print(f"expected compute capability (8, 0), got {cap}")
        sys.exit(1)

def clone_repo(url, target_dir):
    name = url.rstrip("/").split("/")[-1].replace(".git", "")
    path = target_dir / name
    
    if path.exists():
        print(f"pulling {name}...")
        subprocess.run(["git", "-C", str(path), "pull"], check=True)
    else:
        print(f"cloning {name}...")
        subprocess.run(["git", "clone", "--depth", "1", url, str(path)], check=True)
    
    return path

def patch_setup(setup_path):
    print(f"patching {setup_path}...")
    with open(setup_path, "r") as f:
        content = f.read()
    
    pattern = r'os\.environ\["TORCH_CUDA_ARCH_LIST"\]\s*=\s*["\'][^"\']*["\']'
    if re.search(pattern, content):
        content = re.sub(pattern, 'os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"', content)
    else:
        match = re.search(r'(import\s+os\s*\n)', content)
        if match:
            pos = match.end()
            content = content[:pos] + 'os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"\n' + content[pos:]
            
    with open(setup_path, "w") as f:
        f.write(content)

def install_pkg(path):
    print(f"installing {path.name}...")
    env = os.environ.copy()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(path)],
        env=env, check=True
    )

def test():
    print("running smoke test...")
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    
    config = MambaConfig(d_model=256, n_layer=2, vocab_size=1000, ssm_cfg={"d_state": 16})
    model = MambaLMHeadModel(config).cuda()
    
    x = torch.randint(0, 1000, (2, 64), device="cuda")
    with torch.no_grad():
        out = model(x)
    
    assert out.logits.shape == (2, 64, 1000)
    print("test passed")

def main():
    check_prereqs()
    
    temp_dir = Path.cwd() / "temp_mamba_build"
    temp_dir.mkdir(exist_ok=True)
    
    repos = [
        "https://github.com/Dao-AILab/causal-conv1d.git",
        "https://github.com/state-spaces/mamba.git",
    ]
    
    paths = []
    for url in repos:
        paths.append(clone_repo(url, temp_dir))
    
    for p in paths:
        setup = p / "setup.py"
        if setup.exists():
            patch_setup(setup)
        install_pkg(p)
    
    print("pinning transformers...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.39.3"], check=True)
    
    test()

if __name__ == "__main__":
    main()
