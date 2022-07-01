from network.model import StyleGAN2_3D
from utils.render_utils import generate_image_for_fid
from utils.utils import noise, evaluate_in_chunks
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()



model_path = args.model_path
fid_root = "./fid_temp"

model = StyleGAN2_3D.load_from_checkpoint(model_path).cuda(0)
model.eval()
z = noise(100000, model.latent_dim, device=model.device)
samples = evaluate_in_chunks(1000, model.SE, z)
model.av = torch.mean(samples, dim=0, keepdim=True)



mesh = model.generate_mesh(
    ema=True, mc_vol_size=128, level=-0.015, trunc_psi=1.0)
generate_image_for_fid(mesh,fid_root)


