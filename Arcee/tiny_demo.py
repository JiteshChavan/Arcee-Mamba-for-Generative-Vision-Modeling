from create_model import create_model
from dataclasses import dataclass
import torch

C = 3
res = 64 * 8
latent_res = 64
@dataclass
class DNNConfig:
    model="Arcee-Tiny/2"
    ssm_dstate  = 256
    image_size  = res # latent res = 64 x 8 / 8
    num_in_channels  = C
    label_dropout  = -1
    num_classes  = 1
    learn_sigma  = False
    rms_norm = True
    fused_add_norm=True
    scan_type = "Arcee_1"
    num_moe_experts = 0
    gated_linear_unit = False
    routing_mode = "top1"
    is_moe = False
    pe_type = "ape"
    block_type = "normal"
    learnable_pe = True
    drop_path = 0.0
    use_final_norm = False
    use_attn_every_k_layers = -1


model = create_model (DNNConfig).to ('cuda')

x = torch.randn(2, C, latent_res, latent_res).to('cuda')
t = (torch.randn(2) + 1)/2
t = t.to('cuda')
out = model(x, t)
print(out.shape)
loss = out.mean()
loss.backward()

