import torch
from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from data import all_atom
import copy
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch.nn.functional as F


def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    mask_expanded = diffuse_mask[..., None, None]  # expand mask from [B,N] to [B, N, 1, 1]
    trans_1_expanded = trans_1[:, :, None, :] # shape [B, N, 1, 3]
    a = trans_t * mask_expanded
    b = trans_1_expanded * (1 - mask_expanded)
    c = a+b
    return trans_t * mask_expanded + trans_1_expanded * (1 - mask_expanded)

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None, None]
        + rotmats_1[:,:,None,:,:] * (1 - diffuse_mask[..., None, None, None])
    )


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None
        self.num_per_flow = cfg.num_per_flow

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device
    
    def _pad_rotmats(self, rotmats_t):
        '''
            Pad rotmats_t from [B,N,l,3,3], 
                where N is the actual number of residues (and N \leq max_num_res) 
            to [B, max_num_res,l,3,3] with all 0's
        '''
        rotmats_t_padded = F.pad(rotmats_t, (0, 0, 0, 0, 0, 0, 0, self._cfg.max_num_res - rotmats_t.shape[1]), "constant", 0)
        return rotmats_t_padded

    def _pad_trans(self, trans_t):
        '''
            Pad rotmats_t from [B,N,l,3], 
                where N is the actual number of residues (and N \leq max_num_res) 
            to [B, max_num_res,l,3] with all 0's
        '''
        trans_t_padded = F.pad(trans_t, (0, 0, 0, 0, 0, self._cfg.max_num_res - trans_t.shape[1]), "constant", 0)
        return trans_t_padded

    def _pad_res_mask(self, res_mask):
        '''
            Pad rotmats from [B,N],
                where N is the actual number of residues (and N \leq max_num_res) 
            to [B, max_num_res] with all 1's

            Note: pad with 1's not 0's
        '''
        res_mask_padded = F.pad(res_mask, (0, self._cfg.max_num_res - res_mask.shape[1]), mode='constant', value=1)
        return res_mask_padded

    def sample_t(self, num_batch):
        t = torch.linspace(0,1,self.num_per_flow).to(self._device)
        # print("Sample t shape: {}".format(t.shape))
        return t

    def _corrupt_trans(self, trans_1, t, res_mask):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        t_expanded = t.reshape(1,1,self.num_per_flow,1) # dimensions correspond to [B,N,l,1], we let B and N be 1 to let torch broadcase
        trans_0_expanded = trans_0[:, :, np.newaxis, :]
        trans_1_expanded = trans_1[:, :, np.newaxis, :]
        trans_t = (1 - t_expanded) * trans_0_expanded + t_expanded * trans_1_expanded # trans_t has shape [B, N, l, 3]
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None, None]
    
    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        ) 
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)
        
        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device) # [1, B*N,3,3]
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3) # [B, N, 3, 3]
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats) # matrix multiplication of corresponding batch, residue, [B,N,3,3]
        
        
        rotmats_t = so3_utils.my_geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None, None]
            + identity[None, None, None] * (1 - res_mask[..., None, None, None])
        ) # mix rotmats_t with identity according to res_mask, but if res_mask is all 1s then nothing happens, [B, N, l, 3, 3]
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(self, batch, pad=True):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, l, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape
        if pad:
            noisy_batch['res_mask'] = self._pad_res_mask(res_mask)
        noisy_batch['res_mask'] = noisy_batch['res_mask'][:, None, :].repeat(1, self.num_per_flow ,1)

        # [B, l]
        t = self.sample_t(num_batch) # [l]
        noisy_batch['t'] = t.repeat(num_batch, 1)

        # Apply corruptions
        # trans_t shape [B, l, max_num_res, 3]
        trans_t = self._corrupt_trans(trans_1, t, res_mask)
        if pad:
            trans_t = self._pad_trans(trans_t) # [B, max_num_res, l, 3]
        trans_t = trans_t.permute(0, 2, 1, 3) # swap to [B, l, max_num_res, 3]
        noisy_batch['trans_t'] = trans_t

        # rotmats_t shape [B, l, max_num_res, 3, 3]
        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)
        if pad:
            rotmats_t = self._pad_rotmats(rotmats_t) # [B, max_num_res, l, 3, 3]
        rotmats_t = rotmats_t.permute(0, 2, 1, 3, 4) # [B, l, max_num_res, 3, 3]
        noisy_batch['rotmats_t'] = rotmats_t
        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)

    def sample(
            self,
            num_batch,
            num_res,
            model,
        ):
        
        B, l, N = num_batch, self.num_per_flow, num_res

        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        batch = {
            'res_mask': res_mask,
        }
        trans_0 = trans_0[:, :, None,:] # [B, N, 3] -> [B, N, 1, 3]
        rotmats_0 = rotmats_0[:,:,None,:, :] # [B, N, 3, 3] -> [B, N, 1, 3, 3]
        t = self.sample_t(num_batch)[None, :].repeat(num_batch,1).to(self._device) #[B, l]
        rotmats_0 = self._pad_rotmats(rotmats_0).permute(0, 2, 1, 3, 4) # [B, 1, max_num_res, 3, 3]
        trans_0 = self._pad_trans(trans_0).permute(0, 2, 1, 3) # [B, 1, max_num_res, 3]
        batch['res_mask'] = batch['res_mask'][:,None,:].repeat(1, self.num_per_flow, 1) # [B, l, N]
        batch["trans_t"] = trans_0 
        batch["rotmats_t"] = rotmats_0
        batch['t'] = t
        with torch.no_grad():   
            out = model.generate(batch)
        out["pred_T"]["pred_trans"] = out["pred_T"]["pred_trans"].reshape(B, l, N, 3)
        out["pred_T"]["pred_rotmats"] = out["pred_T"]["pred_rotmats"].reshape(B, l, N, 3, 3)
        protein_trajectory = []
        for i in range(out["pred_T"]["pred_trans"].shape[1]):
            protein_trajectory.append(
                (out["pred_T"]["pred_trans"][:, i, :, :].detach().cpu(), out["pred_T"]["pred_rotmats"][:, i, :, :, :].detach().cpu())
            )
        atom37_traj = all_atom.transrot_to_atom37(protein_trajectory, res_mask)
        
        return atom37_traj, 0, 0
