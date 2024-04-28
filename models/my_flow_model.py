"""Neural network architecture for the flow model."""
import torch
from torch import nn

from models.my_node_embedder import NodeEmbedder
from models.edge_embedder import EdgeEmbedder
from models import ipa_pytorch
from data import utils as du


class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_embedder = NodeEmbedder(model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

    def forward(self, input_feats):
        print("Inside forward function of the model: ")
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        continuous_t = input_feats['t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        print("Node mask shape: {}".format(node_mask.shape))
        print("Edge mask shape: {}".format(edge_mask.shape))
        print("Translation shape: {}".format(trans_t.shape))
        print("Rotation matrix shape: {}".format(rotmats_t.shape))

        # Initialize node and edge embeddings
        init_node_embed = self.node_embedder(continuous_t, node_mask)
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
        init_edge_embed = self.edge_embedder(
            init_node_embed, trans_t, trans_sc, edge_mask)

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t,)
        print("Rigid rotmat shape: {}".format(curr_rigids._rots._rot_mats.shape))
        print("Rigit translation shape: {}".format(curr_rigids._trans.shape))

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        print("Node embed shape: {}".format(node_embed.shape))
        print("Edge embed shape: {}".format(edge_embed.shape))
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            print("IPA embed shape: {}".format(ipa_embed.shape))
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            print("Inside for loop: Node embed shape: {}".format(node_embed.shape))
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None]) # rigid_update are 6D vectors representing the quaternion (3D) and the translation vector (3D) 
            
            print("rigid_update shape: {}".format(rigid_update.shape))
            
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None]) # update the rotation matrices and translation vectors by the quaternion

            print("Rigid rotmat shape: {}".format(curr_rigids._rots._quats.shape))
            print("Rigit translation shape: {}".format(curr_rigids._trans.shape))

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
                print("Inside for loop: Edge embed shape: {}".format(edge_embed.shape))

        curr_rigids = self.rigids_nm_to_ang(curr_rigids) # from nanometer to angstorm, change of units
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
        }
