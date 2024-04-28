from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from analysis import metrics 
from analysis import utils as au
from models.together_model import ProteinVAELLMmodel
from models import utils as mu
from data.my_interpolant import Interpolant 
from data import utils as du
from data import all_atom
from data import so3_utils
from data import residue_constants
from experiments import utils as eu
from pytorch_lightning.loggers.wandb import WandbLogger
import torch.nn.functional as F



class FlowModule(LightningModule):

    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = ProteinVAELLMmodel(cfg)
        self.model.attach_backward_hooks()

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()
        
    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        # print(noisy_batch['res_mask'].shape)
        B, l, num_res = noisy_batch['res_mask'].shape
        
        loss_mask = noisy_batch['res_mask'].reshape(B*l, num_res)
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        

        # Ground truth labels
        gt_trans = noisy_batch['trans_t']
        gt_rotmats = noisy_batch['rotmats_t'] 

        # Model output predictions.
        framediff_out = self.model(noisy_batch)
        pred_trans = framediff_out["pred_T"]['pred_trans'].reshape(B, l, 128, 3)
        pred_rotmats = framediff_out["pred_T"]['pred_rotmats'].reshape(B, l, 128, 3, 3)
        
        # Shift for CausalLM
        shifted_pred_trans  = pred_trans[:, :-1, :, :]
        shifted_pred_rotmats = pred_rotmats[:, :-1, :, :, :]
        shifted_gt_trans = gt_trans[:, 1:, :, :]
        shifted_gt_rotmats = gt_rotmats[:, 1:, :, :, :]

        # Reshape back to [B*(l-1), 128, *]
        flat_shifted_pred_trans = shifted_pred_trans.reshape(B*(l-1), self._data_cfg.dataset.max_num_res, 3)
        flat_shifted_pred_rotmats = shifted_pred_rotmats.reshape(B*(l-1), self._data_cfg.dataset.max_num_res, 3, 3)

        flat_shifted_gt_trans = shifted_gt_trans.reshape(B*(l-1), self._data_cfg.dataset.max_num_res, 3)
        flat_shifted_gt_rotmats = shifted_gt_rotmats.reshape(B*(l-1), self._data_cfg.dataset.max_num_res, 3, 3)

        # Timestep used for normalization.
        t = noisy_batch['t'].reshape(B, l, 1)[:, 1:, :].reshape(-1,1) # We throw away the first time points
        norm_scale = 1 - torch.min(
            t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        

        # Backbone atom loss
        gt_bb_atoms = all_atom.to_atom37(flat_shifted_gt_trans, flat_shifted_gt_rotmats)[:, :, :3] 
        pred_bb_atoms = all_atom.to_atom37(flat_shifted_pred_trans, flat_shifted_pred_rotmats)[:, :, :3]

        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
                
        
        loss_denom = torch.sum(loss_mask, dim=-1, dtype=torch.float).mean() * 3 # Added a mean here, this doesn'y matter since our mask is all 1's
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2,
            dim=(-1, -2, -3)
        ) / loss_denom

        # Pairwise distance loss
        num_batch = gt_bb_atoms.shape[0]
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))[B:,:,:] # change the shape because we shifted tokens, all entries of loss masks are 1 so don't matter, we throw away B tokens
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))[B:,:,:] # change the shape because we shifted tokens, all entries of loss masks are 1 so don't matter
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)

        auxiliary_loss = (bb_atom_loss + dist_mat_loss) * (
            t[:, 0]> training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        
        auxiliary_loss = auxiliary_loss.mean()
        kl_div = (1 + 2 * framediff_out["vae_log_sigma"] - framediff_out["vae_mu"].pow(2) - framediff_out["vae_log_sigma"].exp().pow(2))[:, :-1, :] # Throw away the last mu and sigma because we're not using it to predict
        kl_div = - 0.5 * kl_div.sum(dim=-1).mean()
        mse_loss = F.mse_loss(flat_shifted_pred_trans, flat_shifted_gt_trans) + F.mse_loss(flat_shifted_pred_rotmats, flat_shifted_gt_rotmats)

        return {         "bb_atom_loss": bb_atom_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "kl_div": kl_div,
            "mse_loss": mse_loss
        }

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch['res_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        
        samples = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
        )[0][-1].numpy()

        batch_metrics = []
        for i in range(num_batch):

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos,
                os.path.join(
                    self._sample_write_dir,
                    f'sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'),
                no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_idx = residue_constants.atom_order['CA']
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)
        
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name,metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = model_sc['pred_T']['pred_trans']
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch['res_mask'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # # Losses to track. Stratified across t.
        # t = torch.squeeze(noisy_batch['t'])
        # self._log_scalar(
        #     "train/t",
        #     np.mean(du.to_numpy(t)),
        #     prog_bar=False, batch_size=num_batch)
        # for loss_name, loss_dict in batch_losses.items():
        #     stratified_losses = mu.t_stratified_loss(
        #         t, loss_dict, loss_name=loss_name)
        #     for k,v in stratified_losses.items():
        #         self._log_scalar(
        #             f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = total_losses["mse_loss"] + total_losses["auxiliary_loss"] + total_losses["kl_div"]
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )

    def predict_step(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()
        diffuse_mask = torch.ones(1, sample_length)
        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return
        atom37_traj, model_traj, _ = interpolant.sample(
            1, sample_length, self.model
        )

        os.makedirs(sample_dir, exist_ok=True)
        bb_traj = du.to_numpy(torch.concat(atom37_traj, dim=0))
        _ = eu.save_traj(
            bb_traj[-1],
            bb_traj,
            np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
            du.to_numpy(diffuse_mask)[0],
            output_dir=sample_dir,
        )
