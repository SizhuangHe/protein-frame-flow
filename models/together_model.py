import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config



from models.dev_flow_model import FlowModel
from models.vaellm_model import VAE_GPT2
import openfold.utils.rigid_utils as ru



class ProteinVAELLMmodel(nn.Module):
    def __init__(self, cfg):
        super(ProteinVAELLMmodel, self).__init__()
        self._cfg = cfg
        self.num_per_flow = self._cfg.interpolant.num_per_flow
        self._cfg_model = cfg.model
        self._cfg_llm = cfg.model.llm

        self.framediff_model = FlowModel(self._cfg_model)

        # gpt2 = GPT2Model.from_pretrained('gpt2')
        gpt2 = GPT2Model(GPT2Config())
        gpt2.wte=None
        
        self.vaellm_model = VAE_GPT2(base_model=gpt2, emb_dim=self._cfg_llm.emb_dim, z_dim=self._cfg_llm.z_dim)
        # self.vaellm_model.attach_backward_hooks()

    def _get_bb_reps(self, noisy_batch):
        '''
            This function prepares the trajectories into representations for the LLM.
        '''
        # Convert rotmats to quaternions (4D)
        quats = ru.rot_to_quat(noisy_batch['rotmats_t'])

        # Reduce quaternions to 3D
        condition = quats[..., 0] < 0
        quats[condition] *= -1
        quats = quats[..., 1:] # [B, l, N, 3]
        
        # Concatenate with translation vectors
        concat_quats_trans = torch.cat((quats, noisy_batch['trans_t']), dim=-1) # [B, l, N, 6]

        # Concatenate residues to form backbones
        backbone = concat_quats_trans.reshape(*concat_quats_trans.shape[:-2], -1) # [B, l, 6N]

        return backbone

    def _recover_rots_trans(self, z):
        '''
            This function recovers the rotation matrices and translation vectors from the vector representation.

            Parameters
            ----------
            z:          Tensor, [B, l, max_num_res, 6]

            Returns
            -------
            recovered_rots:
                Tensor, [B, l, max_num_res, 3, 3]
            recovered_trans:
                Tensor, [B, l, max_num_res, 3]
        '''
        B, l = z.shape[0], z.shape[1]
        # print("z shape: ", z.shape)
        recovered_reduced_quats = z[:, :, :, :3]
        recovered_trans = z[:, :, :, 3:]
        ones = torch.ones(B, l, z.shape[2], 1).to(recovered_trans.device) # TODO will need to change here
        recovered_quats = torch.cat((ones, recovered_reduced_quats), dim=-1)
        recovered_quats = F.normalize(recovered_quats, p=2, dim=-1) # normalize (1,x,y,z) as AlphaFold did
        recovered_rots = ru.quat_to_rot(recovered_quats)
        return recovered_rots, recovered_trans


    def forward(self, noisy_batch):
        backbone = self._get_bb_reps(noisy_batch)

        llm_out = self.vaellm_model(backbone)

        # Reshaping outputs from llm_out to FrameDiff happy formats
        # The main idea is we collapse minibatch dimension and sequence dimension into one
        # so that FrameDiff thinks our batch size is B * l instead of B
        # This is OK because all computations only happen within one protein backbone anyway
        # print(noisy_batch['rotmats_t'].shape)
        B, l, N, _, _ = noisy_batch['rotmats_t'].shape
        
        z = llm_out["z_sampled"].reshape(B, l, 128, -1) # TODO will need to change here
        
        # Recover rotation matrices and translation vectors
        recovered_rots, recovered_trans = self._recover_rots_trans(z)
        
        # Process shapes
        noisy_batch['processed_rotmats_t'] = recovered_rots
        noisy_batch['processed_trans_t'] = recovered_trans
        noisy_batch['processed_rotmats_t'] = noisy_batch['processed_rotmats_t'].reshape(B*l, N, 3, 3)
        noisy_batch['processed_trans_t'] = noisy_batch['processed_trans_t'].reshape(B*l, N, 3)
        noisy_batch['t'] = noisy_batch['t'].reshape(B*l, 1)
        noisy_batch['res_mask'] = noisy_batch['res_mask'].reshape(B*l, -1)

        framediff_out = self.framediff_model(noisy_batch)

        noisy_batch['res_mask'] = noisy_batch['res_mask'].reshape(B, l, -1)

        return {
            "pred_T": framediff_out,
            "vae_mu": llm_out["mu"],
            "vae_log_sigma": llm_out["log_sigma"]
        }
    
    def generate(self, noisy_batch, T=1):
        backbone = self._get_bb_reps(noisy_batch)
        llm_out = self.vaellm_model.generate(backbone, temperature=T, max_length=self.num_per_flow)
        B = noisy_batch["rotmats_t"].shape[0]
        l = self.num_per_flow
        # print("llm out shape: ", llm_out.shape)
        z = llm_out.reshape(B, l, 128, -1) # TODO will need to change here
        # print("reshaped: ", z.shape)
        N = noisy_batch['res_mask'].shape[2]

        z = z[:, :, :N, :] # We only keep the required number of residues

        recovered_rots, recovered_trans = self._recover_rots_trans(z)

        # Process shapes
        noisy_batch['processed_rotmats_t'] = recovered_rots
        noisy_batch['processed_trans_t'] = recovered_trans
        noisy_batch['processed_rotmats_t'] = noisy_batch['processed_rotmats_t'].reshape(B*l, N, 3, 3)
        noisy_batch['processed_trans_t'] = noisy_batch['processed_trans_t'].reshape(B*l, N, 3)
        noisy_batch['t'] = noisy_batch['t'].reshape(B*l, 1)
        noisy_batch['res_mask'] = noisy_batch['res_mask'].reshape(B*l, -1)

        framediff_out = self.framediff_model(noisy_batch)

        noisy_batch['res_mask'] = noisy_batch['res_mask'].reshape(B, l, -1)
        
        return framediff_out
    
    def attach_backward_hooks(self):
        """ Attach full backward hooks to all layers to check for NaNs in gradients. """
        for layer in self.modules():
            layer.register_full_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        input_nan_detected = False
        output_nan_detected = False

        for idx, grad in enumerate(grad_input):
            if grad is not None and torch.isnan(grad).any():
                print(f"NaN detected in full gradient input {idx} of {module.__class__.__name__}")
                input_nan_detected = True

        for idx, grad in enumerate(grad_output):
            if grad is not None and torch.isnan(grad).any():
                print(f"NaN detected in full gradient output {idx} of {module.__class__.__name__}")
                output_nan_detected = True

        if not input_nan_detected:
            print(f"No NaNs detected in any gradient inputs of {module.__class__.__name__}")
        if not output_nan_detected:
            print(f"No NaNs detected in any gradient outputs of {module.__class__.__name__}")


