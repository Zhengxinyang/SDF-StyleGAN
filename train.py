import fire
import os
from network.model import StyleGAN2_3D
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import loggers as pl_loggers
from utils.utils import ensure_directory, run


def train_from_folder(
    data,
    results_dir='./runs',
    name='default',
    G_vol_size=32,
    D_vol_size=32,
    fine_D_vol_size=128,
    latent_dim=512,
    style_depth=8,
    G_feature_dim=16,
    mlp_network_depth=2,
    mlp_latent_dim=128,
    network_capacity=16,
    G_fmap_max=256,
    D_fmap_max=256,
    surface_level=-0.015,
    batch_size=4,
    learning_rate=1e-4,
    lr_mlp=0.1,
    global_D_lr_mul=1.5,
    local_D_lr_mul=1.5,
    pl_loss_weight=2.0,
    local_patch_loss_weight=1,
    use_global_normal=True,
    apply_gradient_normalization=True,
    use_patch=True,
    local_use_sdf=True,
    local_use_normal=True,
    local_patch_res=16,
    local_patch_number=16,
    local_patch_center_candidate_number=4096,
    r1_gamma=10,
    training_epoch=200,
    mixed_prob=0.9,
    new=False,
    continue_training=True,
    debug=False,
    verbose=False,
    adaptive_local=True,
    offset_center=False,
    save_image=False,
    seed=777
):
  results_dir = results_dir + "/" + name
  ensure_directory(results_dir)

  if continue_training:
    new = False
  if debug and new:
    new = True

  if new:
    run(f"rm -rf {results_dir}/*")

  model_args = dict(
      results_dir=results_dir,
      data=data,
      batch_size=batch_size,
      latent_dim=latent_dim,
      style_depth=style_depth,
      surface_level=surface_level,
      G_feature_dim=G_feature_dim,
      G_vol_size=G_vol_size,
      global_D_lr_mul=global_D_lr_mul,
      local_D_lr_mul=local_D_lr_mul,
      D_vol_size=D_vol_size,
      fine_D_vol_size=fine_D_vol_size,
      use_patch=use_patch,
      use_global_normal=use_global_normal,
      local_use_sdf=local_use_sdf,
      local_use_normal=local_use_normal,
      local_patch_res=local_patch_res,
      local_patch_number=local_patch_number,
      apply_gradient_normalization=apply_gradient_normalization,
      local_patch_center_candidate_number=local_patch_center_candidate_number,
      mlp_network_depth=mlp_network_depth,
      mlp_latent_dim=mlp_latent_dim,
      network_capacity=network_capacity,
      G_fmap_max=G_fmap_max,
      D_fmap_max=D_fmap_max,
      learning_rate=learning_rate,
      lr_mlp=lr_mlp,
      pl_loss_weight=pl_loss_weight,
      local_patch_loss_weight=local_patch_loss_weight,
      r1_gamma=r1_gamma,
      mixed_prob=mixed_prob,
      view_index=2,
      verbose=verbose,
      save_image=save_image,
      adaptive_local=adaptive_local,
      offset_center=offset_center,
      training_epoch=training_epoch
  )

  seed_everything(seed)

  model = StyleGAN2_3D(**model_args)

  log_dir = results_dir

  tb_logger = pl_loggers.TensorBoardLogger(
      save_dir=log_dir,
      version=None,
      name='logs',
      default_hp_metric=False
  )

  checkpoint_callback = ModelCheckpoint(
      monitor="current_epoch",
      dirpath=results_dir,
      filename="{epoch:02d}",
      save_top_k=40,
      save_last=True,
      every_n_epochs=10,
      mode="max",
  )

  if debug:
    if continue_training and os.path.exists(results_dir + '/last.ckpt'):
      trainer = Trainer(devices=-1,
                        accelerator="gpu",
                        strategy=DDPPlugin(find_unused_parameters=False),
                        logger=tb_logger,
                        max_epochs=training_epoch,
                        log_every_n_steps=1,
                        callbacks=[checkpoint_callback],
                        overfit_batches=0.005,
                        resume_from_checkpoint=results_dir + '/last.ckpt')
    else:
      trainer = Trainer(devices=-1,
                        accelerator="gpu",
                        strategy=DDPPlugin(find_unused_parameters=False),
                        logger=tb_logger,
                        max_epochs=training_epoch,
                        log_every_n_steps=1,
                        callbacks=[checkpoint_callback],
                        overfit_batches=0.005)
  else:
    trainer = Trainer(devices=-1,
                      accelerator="gpu",
                      strategy=DDPPlugin(find_unused_parameters=False),
                      logger=tb_logger,
                      max_epochs=training_epoch,
                      log_every_n_steps=1,
                      callbacks=[checkpoint_callback])

  trainer.fit(model)


if __name__ == '__main__':
  fire.Fire(train_from_folder)
