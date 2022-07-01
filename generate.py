import numpy as np
import math
from network.model import StyleGAN2_3D
from pytorch_lightning import seed_everything
import torchvision
from utils.utils import ensure_directory, str2bool
from tqdm import tqdm
from utils.utils import noise, evaluate_in_chunks, scale_to_unit_sphere, volume_noise, process_sdf, linear_slerp
import torch
from utils.render.render import render_mesh

def generate(
    model_path,
    output_path="./outputs",
    ema=True,
    level=0.03,
    mc_vol_size =64,
    num_generate=36,
    trunc_psi=0.75,
    random_index=False,
    render_resolution=1024,
    deterministic=False
):
  model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
  model = StyleGAN2_3D.load_from_checkpoint(model_path).cuda(0)
  model.eval()
  if deterministic:
      seed_everything(777)
  print("calculate self.av")
  z = noise(100000, model.latent_dim, device=model.device)
  if ema:
    samples = evaluate_in_chunks(1000, model.SE, z)
    model.av = torch.mean(samples, dim=0, keepdim=True)
  else:
    samples = evaluate_in_chunks(1000, model.S, z)
    model.av = torch.mean(samples, dim=0, keepdim=True)

  print("start generation")
  images = []
  for i in tqdm(range(num_generate)):
    mesh =  model.generate_mesh(
      ema=ema, mc_vol_size=mc_vol_size, level=level, trunc_psi=trunc_psi)
    while mesh is None:
        mesh = model.generate_mesh(
      ema=ema, mc_vol_size=mc_vol_size, level=level, trunc_psi=trunc_psi)
    mesh = scale_to_unit_sphere(mesh)
    if random_index:
      index = np.random.randint(20)
    else:
      index = 0
    
    image = render_mesh(mesh, index=index, resolution=render_resolution)/255
    image = torch.from_numpy(image.copy()).permute(2, 0, 1)
    images.append(image)

  name = f'generate_{model_name}_{model_id}_{ema}_{level}_{trunc_psi}'
  torchvision.utils.save_image(
      images, output_path + f"/{name}.png", nrow=min(math.ceil(np.sqrt(num_generate)), 15))


def generate_mesh(
    model_path,
    output_path="./outputs",
    ema=True,
    level=0.03,
    trunc_psi=0.75,
    mc_vol_size=64,
    num_generate=10,
    deterministic=False
):
  model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
  model = StyleGAN2_3D.load_from_checkpoint(model_path).cuda(0)
  model.eval()
  if deterministic:
      seed_everything(777)
  print("calculate self.av")
  z = noise(100000, model.latent_dim, device=model.device)
  if ema:
    samples = evaluate_in_chunks(1000, model.SE, z)
    model.av = torch.mean(samples, dim=0, keepdim=True)
  else:
    samples = evaluate_in_chunks(1000, model.S, z)
    model.av = torch.mean(samples, dim=0, keepdim=True)
  print("start generation")
  for i in tqdm(range(num_generate)):
    mesh = model.generate_mesh(
        ema=ema, mc_vol_size=mc_vol_size, level=level, trunc_psi=trunc_psi)
    mesh = scale_to_unit_sphere(mesh)
    name = f'/generate_{model_name}_{model_id}_{ema}_{level}_{mc_vol_size}_{trunc_psi}_{i}.obj'
    mesh.export(output_path + name)


def generate_interpolation(
    model_path,
    output_path="./outputs",
    ema=True,
    level=0.03,
    vol_size=64,
    render_resolution=1024,
    interpolation_num_steps=16,
    trunc_psi=0.75,
    view_index=5,
    rows=1,
    deterministic=False
):
  model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
  model = StyleGAN2_3D.load_from_checkpoint(model_path).cuda(0)
  model.eval()
  if deterministic:
    seed_everything(777)
  print("calculate self.av")
  z = noise(100000, model.latent_dim, device=model.device)
  if ema:
    samples = evaluate_in_chunks(1000, model.SE, z)
    model.av = torch.mean(samples, dim=0, keepdim=True)
  else:
    samples = evaluate_in_chunks(1000, model.S, z)
    model.av = torch.mean(samples, dim=0, keepdim=True)
  print("start generation")
  n = volume_noise(1, model.G_vol_size, device=model.device)

  interpolation_num_steps = 40
  for i in tqdm(range(rows)):
    images = []
    latents_low = noise(1, model.latent_dim, device=model.device)
    latents_high = noise(1, model.latent_dim, device=model.device)

    ratios = torch.linspace(0., 1., interpolation_num_steps)
    for j in range(interpolation_num_steps):
      ratio = ratios[j]
      interp_latents = linear_slerp(ratio, latents_low, latents_high)
      latents = [(interp_latents, model.num_layers)]
      points = model.get_voxel_coordinates(resolution=vol_size)
      
      generated_voxels = model.generate_truncated(
          model.SE, model.GE, latents, n, trunc_psi=trunc_psi)
      sdf = model.ME(points, generated_voxels).view(
          vol_size, vol_size, vol_size) - level
      
      mesh = scale_to_unit_sphere(
          process_sdf(sdf.cpu().numpy(), level=level))
      image = render_mesh(mesh, index=view_index, resolution=render_resolution)/255
      image = torch.from_numpy(image.copy()).permute(2, 0, 1)
      images.append(image)
      
  name = f'interpolation_{rows}_{interpolation_num_steps}_{model_name}_{model_id}_{ema}_{level}_{trunc_psi}_{view_index}'
  torchvision.utils.save_image(
    images, output_path + f"/{name}.png", nrow=interpolation_num_steps)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='generate something')

  parser.add_argument("--generate_method", type=str, default='generate',
                      help="please choose :\n \
                       1. 'generate' \n \
                       2. 'generate_interpolation' \n \
                       3. 'generate_mesh'")

  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--output_path", type=str, default="./outputs")
  parser.add_argument("--ema", type=str2bool, default=True)
  parser.add_argument("--level", type=float, default=-0.015)
  parser.add_argument("--num_generate", type=int, default=100)
  parser.add_argument("--interpolation_num_steps", type=int, default=10)
  parser.add_argument("--interpolation_rows", type=int, default=10)
  parser.add_argument("--trunc_psi", type=float, default=1.0)
  parser.add_argument("--shading_method", type=str, default="normal")
  parser.add_argument("--view_index", type=int, default=1)
  parser.add_argument("--col_num", type=int, default=10)
  parser.add_argument("--mc_vol_size", type=int, default=128)
  parser.add_argument("--random_index", type=str2bool, default=False)
  parser.add_argument("--render_resolution", type=int, default=1024)
  parser.add_argument("--deterministic", type=str2bool, default=False)

  args = parser.parse_args()
  method = (args.generate_method).lower()
  ensure_directory(args.output_path)
  if method == "generate":
    generate(model_path=args.model_path, output_path=args.output_path, ema=args.ema, level=args.level, mc_vol_size=args.mc_vol_size, num_generate=args.num_generate,
             trunc_psi=args.trunc_psi,random_index=args.random_index, render_resolution=args.render_resolution, deterministic=args.deterministic)
  elif method == "generate_interpolation":
    generate_interpolation(model_path=args.model_path, output_path=args.output_path, ema=args.ema, level=args.level, vol_size=args.mc_vol_size, interpolation_num_steps=args.interpolation_num_steps,
                           trunc_psi=args.trunc_psi, view_index=args.view_index, rows=args.interpolation_rows,render_resolution=args.render_resolution, deterministic=args.deterministic)
  elif method == "generate_mesh":
    generate_mesh(model_path=args.model_path, output_path=args.output_path, ema=args.ema,
                  level=args.level, trunc_psi=args.trunc_psi, mc_vol_size=args.mc_vol_size,num_generate=args.num_generate, deterministic=args.deterministic)
  else:
    raise NotImplementedError
