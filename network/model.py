import os
import torch
import torch.nn as nn
from torch.optim import Adam
from pytorch_lightning import LightningModule
from network.dataloader import sdfDataset
from torch.utils.data import DataLoader
import numpy as np
from utils.feture_interpolation import feature_interpolation_trilinear
from utils.utils import *
from random import random, sample
from utils.utils import scale_to_unit_sphere
from .discriminator import Discriminator_3D
from .generator import Generator, MLP_Net, StyleVectorizer
from .loss import simple_gradient_penalty, gen_stylegan_loss, dis_stylegan_loss, calc_pl_lengths_3d

from .custom_layer import EMA


class StyleGAN2_3D(LightningModule):
  def __init__(
      self,
      results_dir=None,
      data=None,
      G_vol_size=32,
      D_vol_size=32,
      fine_D_vol_size=128,
      latent_dim=128,
      style_depth=8,
      G_feature_dim=32,
      mlp_network_depth=4,
      mlp_latent_dim=128,
      network_capacity=16,
      G_fmap_max=256,
      D_fmap_max=256,
      batch_size=16,
      learning_rate=1e-4,
      apply_gradient_normalization=False,
      lr_mlp=0.1,
      surface_level=-0.005,
      global_D_lr_mul=1.5,
      local_D_lr_mul=1.5,
      pl_loss_weight=0.1,
      local_patch_loss_weight=2,
      local_patch_center_candidate_number=4096,
      use_patch=True,
      use_global_normal=False,
      local_use_sdf=False,
      local_use_normal=True,
      local_patch_res=16,
      local_patch_number=16,
      r1_gamma=10,
      mixed_prob=0.9,
      view_index=2,
      verbose=False,
      save_image=True,
      adaptive_local=True,
      offset_center=False,
      training_epoch=100,
  ):
    super().__init__()
    self.automatic_optimization = False
    self.save_hyperparameters()
    self.results_dir = results_dir
    self.save_image = save_image

    self.use_global_normal = use_global_normal

    self.use_patch = use_patch
    self.apply_gradient_normalization = apply_gradient_normalization

    self.data = data
    self.batch_size = batch_size
    
    self.local_patch_number = local_patch_number
    self.local_patch_res = local_patch_res
    self.local_patch_size = local_patch_res / fine_D_vol_size  
    self.local_use_sdf = local_use_sdf
    self.local_use_normal = local_use_normal
    self.local_patch_center_candidate_number = local_patch_center_candidate_number
    self.offset_center = offset_center
    assert self.local_patch_number * 4 <= self.local_patch_center_candidate_number

    self.latent_dim = latent_dim  
    self.G_vol_size = G_vol_size
    self.D_vol_size = D_vol_size
    self.fine_D_vol_size = fine_D_vol_size

    self.lr = learning_rate
    self.global_D_lr_mul = global_D_lr_mul
    self.local_D_lr_mul = local_D_lr_mul

    self.r1_gamma = r1_gamma  
    self.pl_loss_weight = pl_loss_weight  

    self.adaptive_local = adaptive_local
    self.local_patch_loss_weight = local_patch_loss_weight


    self.mixed_prob = mixed_prob
    self.av = None
    self.pl_mean = None

    self.surface_level = surface_level
    self.view_index = view_index
    self.training_epoch = training_epoch
    self.verbose = verbose


    self.S = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
    self.G = Generator(G_vol_size, latent_dim, network_capacity, fmap_max=G_fmap_max,
                       feature_dim=G_feature_dim)
    self.M = MLP_Net(mlp_network_depth=mlp_network_depth, feature_dim=G_feature_dim,
                     latent_dim=mlp_latent_dim)

    self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
    self.GE = Generator(G_vol_size, latent_dim, network_capacity, fmap_max=G_fmap_max,
                        feature_dim=G_feature_dim)
    self.ME = MLP_Net(mlp_network_depth=mlp_network_depth, feature_dim=G_feature_dim,
                      latent_dim=mlp_latent_dim)

    set_requires_grad(self.SE, False)
    set_requires_grad(self.GE, False)
    set_requires_grad(self.ME, False)

    if use_global_normal:
      D_feature_dim = 4
    else:
      D_feature_dim = 1

    self.global_D = Discriminator_3D(D_vol_size, fmap_max=D_fmap_max, network_capacity=network_capacity,
                                     use_feature=True, feature_dim=D_feature_dim)

    if self.use_patch:
      D_feature_dim = self.init_patch_sataus()
      self.local_D = Discriminator_3D(local_patch_res, fmap_max=D_fmap_max, network_capacity=network_capacity,
                                      use_feature=True, feature_dim=D_feature_dim)

    self.generator_params = list(
        self.S.parameters()) + list(self.G.parameters()) + list(self.M.parameters())

    self.mse_loss = nn.MSELoss()
    self.ema_updater = EMA(0.995)
    self.pl_length_ema = EMA(0.95)

    self._init_weights()
    self.reset_parameter_averaging()
    self.num_layers = self.G.num_layers

  def valid_gradient(self, model):

    for param in model.parameters():
      if param.grad is not None:
        valid_gradients_nan = not torch.isnan(param.grad).any()
        valid_gradients_inf = not torch.isinf(param.grad).any()
        if not valid_gradients_inf:
          print("inf detected!!!")
          return False
        if not valid_gradients_nan:
          print("nan detected!!!")
          return False
    return True

  def init_patch_sataus(self):
    if self.local_use_sdf and self.local_use_normal:
      D_feature_dim = 4
      status = 2
    elif self.local_use_sdf and not self.local_use_normal:
      D_feature_dim = 1
      status = 0
    elif not self.local_use_sdf and self.local_use_normal:
      D_feature_dim = 3
      status = 1
    self.patch_staus = status
    return D_feature_dim

  def _init_weights(self):
    for m in self.modules():
      if type(m) in {nn.Conv3d, nn.Linear}:
        nn.init.kaiming_normal_(
            m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

    for block in self.G.blocks:
      nn.init.zeros_(block.to_noise1.weight)
      nn.init.zeros_(block.to_noise2.weight)
      nn.init.zeros_(block.to_noise1.bias)
      nn.init.zeros_(block.to_noise2.bias)

  def update_EMA(self):
    def update_moving_average(ma_model, current_model):
      for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

    update_moving_average(self.SE, self.S)
    update_moving_average(self.GE, self.G)
    update_moving_average(self.ME, self.M)

  def reset_parameter_averaging(self):
    self.SE.load_state_dict(self.S.state_dict())
    self.GE.load_state_dict(self.G.state_dict())
    self.ME.load_state_dict(self.M.state_dict())

  def train_dataloader(self):
    fine_dataset = sdfDataset(self.data)
    fine_dataloader = DataLoader(
        fine_dataset, num_workers=os.cpu_count(), batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    return fine_dataloader

  def get_voxel_coordinates(self, resolution=32, size=1, center=None):
    if center is None:
      center = (0, 0, 0)
    elif type(center) == int:
      center = (center, center, center)
    else:
      try:
        center = center.detach().cpu().numpy()
      except Exception as e:
        print(str(e))

    points = np.meshgrid(
        np.linspace(center[0] - size, center[0] + size, resolution),
        np.linspace(center[1] - size, center[1] + size, resolution),
        np.linspace(center[2] - size, center[2] + size, resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose()
    return torch.tensor(points, dtype=torch.float32, device=self.device)


  def get_patch_points(self, patch_centers, patch_number, patch_res, patch_size):
    n = patch_res**3
    patch_points = torch.zeros(
        self.batch_size, n * patch_number, 3, device=self.device)
    for i in range(self.batch_size):
      for j in range(patch_number):
        patch_points[i, n*j:n*(j+1)] = self.get_voxel_coordinates(
            resolution=patch_res, size=patch_size, center=patch_centers[i, j])
    return patch_points

  def vaild_mask(self, mask):
    for i in range(self.batch_size):
      if torch.count_nonzero(mask[i]) < self.local_patch_number:
        print(f"mask_{i} number:", torch.count_nonzero(mask[i]))
        return False
    return True

  def get_valid_patch_center(self, small_sdf, low_res_sdf_grid):
    patch_centers = get_voxel_coordinates(
        resolution=self.D_vol_size, device=self.device)[small_sdf.indices, :]
    mask = (patch_centers[:, :, 0] <= 1 - self.local_patch_size) & (patch_centers[:, :, 0] >= - 1 + self.local_patch_size) \
        & (patch_centers[:, :, 1] <= 1 - self.local_patch_size) & (patch_centers[:, :, 1] >= - 1 + self.local_patch_size) \
        & (patch_centers[:, :, 2] <= 1 - self.local_patch_size) & (patch_centers[:, :, 2] >= - 1 + self.local_patch_size)
    assert self.vaild_mask(mask)
    
    res = torch.zeros(self.batch_size, self.local_patch_number,
                      3, device=self.device)
    for i in range(self.batch_size):
      candidate = patch_centers[i][mask[i]]
      index = torch.tensor(sample(range(
          candidate.shape[0]), self.local_patch_number), dtype=torch.long, device=self.device)
      res[i] = candidate[index]

    return res

  def get_patch(self, sdf_grid, generated_voxels=None, from_data=False):

    if from_data:
      low_res_sdf_grid = F.interpolate(sdf_grid, size=self.D_vol_size, mode='trilinear',
                                       align_corners=True).view(self.batch_size, -1)
      small_sdf = torch.topk(torch.abs(low_res_sdf_grid - self.surface_level),
                             self.local_patch_center_candidate_number, largest=False, sorted=False)
    else:
      small_sdf = torch.topk(torch.abs(sdf_grid.view(self.batch_size, -1) - self.surface_level),
                             self.local_patch_center_candidate_number, largest=False, sorted=False)

    patch_centers = self.get_valid_patch_center(
        small_sdf, low_res_sdf_grid if from_data else sdf_grid).detach()

    patch_number = self.local_patch_number
    patch_res = self.local_patch_res
    patch_size = self.local_patch_size

    patch_points = self.get_patch_points(
        patch_centers, patch_number=patch_number, patch_res=patch_res, patch_size=patch_size)

    B = self.batch_size * patch_number
    if from_data:
      if self.patch_staus == 0: 
        patch = feature_interpolation_trilinear(
            patch_points, sdf_grid, points_in_batch=True).view(
            B, 1, patch_res, patch_res, patch_res)
      elif self.patch_staus == 1: 
        patch = feature_interpolation_trilinear(
            patch_points, sdf_grid, points_in_batch=True).view(
            B, 1, patch_res, patch_res, patch_res)
        patch = get_grid_normal(patch, bouding_box_length=2*patch_size)
        if self.apply_gradient_normalization:
          patch = patch/(patch.norm(dim=1, keepdim=True) + 1e-8)
      elif self.patch_staus == 2: 
        patch_sdf = feature_interpolation_trilinear(
            patch_points, sdf_grid, points_in_batch=True).view(
            B, 1, patch_res, patch_res, patch_res)
        patch_grad = get_grid_normal(
            patch_sdf, bouding_box_length=2*patch_size)
        if self.apply_gradient_normalization:
          patch_grad = patch_grad/(patch_grad.norm(dim=1, keepdim=True) + 1e-8)
        patch = torch.cat([patch_sdf, patch_grad], dim=1)
    else:
      if self.patch_staus == 0:
        patch = self.M(patch_points, feature_volume=generated_voxels, points_in_batch=True).view(
            B, 1, patch_res, patch_res, patch_res)
      elif self.patch_staus == 1:
        patch = self.M(patch_points, feature_volume=generated_voxels, points_in_batch=True).view(
            B, 1, patch_res, patch_res, patch_res)
        patch = get_grid_normal(patch, bouding_box_length=2*patch_size)
        if self.apply_gradient_normalization:
          patch = patch/(patch.norm(dim=1, keepdim=True) + 1e-8)
      elif self.patch_staus == 2:
        patch_sdf = self.M(patch_points, feature_volume=generated_voxels, points_in_batch=True).view(
            B, 1, patch_res, patch_res, patch_res)
        patch_grad = get_grid_normal(
            patch_sdf, bouding_box_length=2*patch_size)
        if self.apply_gradient_normalization:
          patch_grad = patch_grad/(patch_grad.norm(dim=1, keepdim=True) + 1e-8)
        patch = torch.cat([patch_sdf, patch_grad], dim=1)

    return patch, patch_centers

  def training_step(self, batch, batch_idx):
    batch_size = self.batch_size
    G_vol_size = self.G_vol_size
    D_vol_size = self.D_vol_size
    latent_dim = self.latent_dim
    num_layers = self.num_layers

    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
    style = get_latents_fn(batch_size, num_layers,
                           latent_dim, device=self.device)
    noise = volume_noise(batch_size, G_vol_size, device=self.device)
    style = latent_to_w(self.S, style)
    w_styles = styles_def_to_tensor(style)


    fine_sdf = batch
    coarse_sdf = F.interpolate(fine_sdf, size=D_vol_size, mode='trilinear',
                               align_corners=True)

    generated_voxels = self.G(w_styles, noise)

    points = self.get_voxel_coordinates(resolution=D_vol_size)

    fake_sdf = self.M(points=points, feature_volume=generated_voxels).view(
        batch_size, 1, D_vol_size, D_vol_size, D_vol_size)

    if self.use_global_normal:
      fake_sdf_grad = get_grid_normal(fake_sdf, bouding_box_length=2)
      coarse_sdf_grad = get_grid_normal(coarse_sdf, bouding_box_length=2)

      if self.apply_gradient_normalization:
        fake_sdf_grad = fake_sdf_grad / \
            (fake_sdf_grad.norm(dim=1, keepdim=True) + 1e-8)
        coarse_sdf_grad = coarse_sdf_grad / \
            (coarse_sdf_grad.norm(dim=1, keepdim=True) + 1e-8)
      fake_sdf_input = torch.cat(
          [fake_sdf, fake_sdf_grad], dim=1).detach()
      coarse_sdf_input = torch.cat(
          [coarse_sdf, coarse_sdf_grad], dim=1)
    else:
      fake_sdf_input = fake_sdf.detach()
      coarse_sdf_input = coarse_sdf

    fake_output = self.global_D(fake_sdf_input)
    real_output = self.global_D(coarse_sdf_input)

    self.log("scores/global_real", real_output.mean(), prog_bar=True)
    self.log("scores/global_fake", fake_output.mean(), prog_bar=True)

    global_d_loss = dis_stylegan_loss(real_output, fake_output)

    coarse_sdf_input.requires_grad_()
    global_gp = simple_gradient_penalty(self.global_D, coarse_sdf_input, fake_sdf_input,
                                        self.device) * self.r1_gamma/2
    total_global_d_loss = global_d_loss + global_gp

    self.log("global discriminator loss/total discriminator loss",
             total_global_d_loss.clone().detach().item())
    self.log("global discriminator loss/discriminator loss",
             global_d_loss.clone().detach().item())
    self.log("global discriminator loss/gradient penalty",
             global_gp.clone().detach().item())

    opt = self.optimizers()[1]
    opt.zero_grad()
    self.manual_backward(total_global_d_loss)
    if self.verbose:
      if self.valid_gradient(self.global_D):
        opt.step()
    else:
      opt.step()
    opt.zero_grad()

    if self.use_patch and self.current_epoch > 0:

      fake_patch, fake_centers = self.get_patch(
          sdf_grid=fake_sdf, generated_voxels=generated_voxels, from_data=False)

      fake_patch = fake_patch.detach()
      real_patch, real_centers = self.get_patch(
          sdf_grid=fine_sdf, generated_voxels=None, from_data=True)

      fake_output = self.local_D(fake_patch)
      real_output = self.local_D(real_patch)

      self.log("scores/local_real", real_output.mean(), prog_bar=True)
      self.log("scores/local_fake", fake_output.mean(), prog_bar=True)

      local_d_loss = dis_stylegan_loss(real_output, fake_output)
      real_patch.requires_grad_()

      local_gp = simple_gradient_penalty(self.local_D, real_patch, fake_patch,
                                         self.device) * self.r1_gamma/2

      total_local_d_loss = local_d_loss + local_gp
      self.log("local discriminator loss/total discriminator loss",
               total_local_d_loss.clone().detach().item())
      self.log("local discriminator loss/discriminator loss",
               local_d_loss.clone().detach().item())
      self.log("local discriminator loss/gradient penalty",
               local_gp.clone().detach().item())

      opt = self.optimizers()[2]
      opt.zero_grad()
      self.manual_backward(total_local_d_loss)

      if self.verbose:
        if self.valid_gradient(self.local_D):
          opt.step()
      else:
        opt.step()
      opt.zero_grad()

    # train generator
    total_gen_loss = 0

    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
    style = get_latents_fn(batch_size, num_layers,
                           latent_dim, device=self.device)
    noise = volume_noise(batch_size, G_vol_size, device=self.device)
    style = latent_to_w(self.S, style)
    w_styles = styles_def_to_tensor(style)
    generated_voxels = self.G(w_styles, noise)

    apply_pl_loss = (self.global_step % 50 == 0) 
    if apply_pl_loss:
      pl_lengths = calc_pl_lengths_3d(w_styles, generated_voxels)
      avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())
      if exists(self.pl_mean):
        pl_loss = self.pl_loss_weight * \
            torch.mean(torch.square(pl_lengths - self.pl_mean))
        total_gen_loss += pl_loss
        self.log("generator loss/pl loss", pl_loss.clone().detach().item())

      self.pl_mean = self.pl_length_ema.update_average(
          self.pl_mean, avg_pl_length)

    points = self.get_voxel_coordinates(resolution=D_vol_size)
    sdf = self.M(points=points, feature_volume=generated_voxels).view(
        batch_size, 1, D_vol_size, D_vol_size, D_vol_size)

    if self.use_global_normal:
      sdf_grad = get_grid_normal(sdf, bouding_box_length=2)
      if self.apply_gradient_normalization:
        sdf_grad = sdf_grad/(sdf_grad.norm(dim=1, keepdim=True) + 1e-8)
      sdf_input = torch.cat([sdf, sdf_grad], dim=1)
    else:
      sdf_input = sdf

    fake_global_output = self.global_D(sdf_input)
    g_global_loss = gen_stylegan_loss(fake_global_output, None)
    total_gen_loss += g_global_loss
    self.log("generator loss/global loss",
             g_global_loss.clone().detach().item())

    if self.use_patch and self.current_epoch > 0:
      fake_patch, fake_centers = self.get_patch(
          sdf_grid=sdf, generated_voxels=generated_voxels, from_data=False)

      fake_local_output = self.local_D(fake_patch)
      if self.adaptive_local:
        adaptive_loss_weight = self.local_patch_loss_weight * \
            (2 * min((self.current_epoch + 1)/self.training_epoch, 1))
      else:
        adaptive_loss_weight = self.local_patch_loss_weight

      g_local_loss = adaptive_loss_weight * \
          gen_stylegan_loss(fake_local_output, None)
      total_gen_loss += g_local_loss
      self.log("generator loss/local loss",
               g_local_loss.clone().detach().item())

    self.log("generator loss/total generator loss",
             total_gen_loss.clone().detach().item())
    opt = self.optimizers()[0]
    opt.zero_grad()
    self.manual_backward(total_gen_loss)
    if self.verbose:
      if self.valid_gradient(self.G) and self.valid_gradient(self.S) and self.valid_gradient(self.M):
        opt.step()
    else:
      opt.step()

    opt.zero_grad()

    self.update_EMA()

  def configure_optimizers(self):
    opts = []
    lr = self.lr
    betas = (0.5, 0.9)
    opt_g = Adam(self.generator_params, lr=lr, betas=betas)
    opts.append(opt_g)
    opt_global_d = Adam(self.global_D.parameters(), lr=lr *
                        self.global_D_lr_mul, betas=betas)
    opts.append(opt_global_d)
    if self.use_patch:
      opt_local_d = Adam(self.local_D.parameters(), lr=lr *
                         self.local_D_lr_mul, betas=betas)
      opts.append(opt_local_d)

    return opts, []

  def on_train_epoch_end(self):
    self.log("current_epoch", self.current_epoch)
    return super().on_train_epoch_end()


  @torch.no_grad()
  def generate_feature_volume(self, ema=False, trunc_psi=0.75):
    latents = noise_list(
        1, self.num_layers, self.latent_dim, device=self.device)
    n = volume_noise(1, self.G_vol_size, device=self.device)
    if ema:
      generate_voxels = self.generate_truncated(
          self.SE, self.GE, latents, n, trunc_psi)
    else:
      generate_voxels = self.generate_truncated(
          self.S, self.G, latents, n, trunc_psi)

    return generate_voxels

  @torch.no_grad()
  def generate_sdf_grid(self, ema=False, vol_size=64, trunc_psi=0.75):
    generated_voxels = self.generate_feature_volume(
        ema=ema, trunc_psi=trunc_psi)
    # print(generated_voxels)
    if vol_size <= 128:
      points = self.get_voxel_coordinates(
          resolution=vol_size)
      if ema:
        sdf = self.ME(points, generated_voxels).view(
            vol_size, vol_size, vol_size)
      else:
        sdf = self.M(points, generated_voxels).view(
            vol_size, vol_size, vol_size)
    else:
      assert ema
      dim = vol_size
      dima = 64
      multiplier = int(dim/dima)
      multiplier2 = multiplier*multiplier
      multiplier3 = multiplier*multiplier*multiplier

      aux_x = torch.zeros([dima, dima, dima],
                          device=generated_voxels.device, dtype=torch.long)
      aux_y = torch.zeros([dima, dima, dima],
                          device=generated_voxels.device, dtype=torch.long)
      aux_z = torch.zeros([dima, dima, dima],
                          device=generated_voxels.device, dtype=torch.long)
      for i in range(dima):
        for j in range(dima):
          for k in range(dima):
            aux_x[i, j, k] = i*multiplier
            aux_y[i, j, k] = j*multiplier
            aux_z[i, j, k] = k*multiplier

      coords = torch.zeros([multiplier3, dima, dima, dima, 3],
                           device=generated_voxels.device)
      for i in range(multiplier):
        for j in range(multiplier):
          for k in range(multiplier):
            coords[i*multiplier2+j*multiplier+k, :, :, :, 0] = aux_x+i
            coords[i*multiplier2+j*multiplier+k, :, :, :, 1] = aux_y+j
            coords[i*multiplier2+j*multiplier+k, :, :, :, 2] = aux_z+k
      coords = (coords/(dim - 1)*2.0-1.0).reshape(multiplier3, -1, 3)

      sdf = torch.zeros([dim, dim, dim], device=generated_voxels.device)
      for i in range(multiplier):
        for j in range(multiplier):
          for k in range(multiplier):
            # print(i,j,k)
            minib = i*multiplier2+j*multiplier+k

            model_out = self.ME(coords[minib], generated_voxels).view(
                dima, dima, dima)
            sdf[aux_x+i, aux_y+j, aux_z+k] = model_out

    return sdf


  @torch.no_grad()
  def generate_mesh(self, ema=False, mc_vol_size=32, level=0.03, trunc_psi=0.75):
    sdf = self.generate_sdf_grid(
        ema=ema, vol_size=mc_vol_size, trunc_psi=trunc_psi)
    # print(sdf)
    volume = sdf.detach().cpu().numpy()
    mesh = process_sdf(volume, level=level)
    return mesh

  @torch.no_grad()
  def generate_feature_volume_from_style(self,  w_styles, ema=True, n=None):
    if n is None:
      n = volume_noise(1, self.G_vol_size, device=self.device)
    if ema:
      generated_voxels = evaluate_in_chunks(
          self.batch_size, self.GE, w_styles, n)
    else:
      generated_voxels = evaluate_in_chunks(
          self.batch_size, self.G, w_styles, n)

    return generated_voxels
  
  @torch.no_grad()
  def generate_sdf_grid_from_style(self,  w_styles, level=0.03, vol_size=64, ema=True, n=None):
    if n is None:
      n = volume_noise(1, self.G_vol_size, device=self.device)
    if ema:
      generated_voxels = evaluate_in_chunks(
          self.batch_size, self.GE, w_styles, n)

      points = self.get_voxel_coordinates(resolution=vol_size)
      sdf = self.ME(points, generated_voxels).view(
          vol_size, vol_size, vol_size)
    else:
      generated_voxels = evaluate_in_chunks(
          self.batch_size, self.G, w_styles, n)

      points = self.get_voxel_coordinates(resolution=vol_size)
      sdf = self.M(points, generated_voxels).view(
          vol_size, vol_size, vol_size)

    return sdf - level


  @torch.no_grad()
  def truncate_style(self, tensor, trunc_psi=0.75):
    batch_size = self.batch_size
    latent_dim = self.G.latent_dim

    if not exists(self.av):
      z = noise(10000, latent_dim, device=self.device)
      samples = evaluate_in_chunks(batch_size, self.S, z)
      self.av = torch.mean(samples, axis=0, keepdim=True)
    av_torch = self.av
    tensor = trunc_psi * (tensor - av_torch) + av_torch
    return tensor

  @torch.no_grad()
  def truncate_style_defs(self, w, trunc_psi=0.75):
    w_space = []
    for tensor, num_layers in w:
      tensor = self.truncate_style(tensor, trunc_psi=trunc_psi)
      w_space.append((tensor, num_layers))
    return w_space

  @torch.no_grad()
  def generate_truncated(self, S, G, style, noi, trunc_psi=0.75):
    w = map(lambda t: (S(t[0]), t[1]), style)
    style = self.truncate_style_defs(w, trunc_psi=trunc_psi)
    w_styles = styles_def_to_tensor(style)
    generated_voxel = evaluate_in_chunks(self.batch_size, G, w_styles, noi)

    return generated_voxel
