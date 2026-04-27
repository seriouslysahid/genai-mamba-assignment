# quick sanity check
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

from mamba_ssm import Mamba
m = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).cuda().to(torch.bfloat16)
x = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)
y = m(x)
print("ok", y.shape)
