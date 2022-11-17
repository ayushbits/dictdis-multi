# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
    PointerNet,
    ConsPosiEmb,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerEncoderBase":
        return "TransformerEncoder"
    else:
        return module_name

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class TransformerEncoderBase(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.encoder_layerdrop = cfg.encoder.layerdrop
        self.return_fc = return_fc

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = cfg.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)
        self.consnmt = cfg.consnmt # set to use constrain nmt
        if self.consnmt:
            cfg.max_constraints_number = 1024
            # print('max_cons',args.max_constraints_number)
            self.cons_pos_embed = ConsPosiEmb(cfg.decoder_embed_dim, self.padding_idx)
            self.seg_embed = Embedding(cfg.max_constraints_number, cfg.decoder_embed_dim, self.padding_idx)
            # print('dec_emb_dim',args.decoder_embed_dim)


        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.encoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
        )
        self.num_layers = len(self.layers)

        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(
            cfg, return_fc=self.return_fc
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        fanout_1: Optional[torch.Tensor] = None,
        fanout_n: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        decoder=None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # print('decoder inside txencoder is ', decoder)
        return self.forward_scriptable(
            src_tokens,decoder,fanout_1,fanout_n, src_lengths, return_all_hiddens, token_embeddings
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        decoder=None,
        fanout_1: Optional[torch.Tensor] = None,
        fanout_n: Optional[torch.Tensor] = None,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # print(self.consnmt)
        # print(src_tokens.view(-1))
        # print(torch.cuda.current_device())
        if not self.consnmt or (4 not in src_tokens.view(-1)):
            # print("########################## WHY IS IT ENTERING HERE ##################################")
            # print('src_token',src_tokens.shape)
            # print('src_token full',src_tokens)
            # print('embed_tokens',self.embed_tokens.embedding_dim)
            # print('embed_tokens full',self.embed_tokens)

            # print("decoder separator id",4)
            # print("consnmt",self.consnmt)

            # if(4 in src_tokens.view(-1)):
            #     print("sep exists in src tokens then why is it entering here")

            
            # print('TRUE')
            # print("consnmt", self.consnmt)
            # for i, pp in enumerate(src_tokens.numpy()):
            #     for j in pp:
            #         if (j<0) or (j>18970):
            #             print(j)
            # # for i, pp in enumerate(src_tokens.numpy()):
            # #     sent = pp
            # #     print("src tokens :",sent)
            # #     src_tkn = self.dictionary.string(sent)
            # #     print("source",src_tkn)


            x = self.embed_scale * self.embed_tokens(src_tokens)
            src_x = x
            n_cons = 0
            n_sub_cons = 0
            sep_position = len(src_tokens) 
            src_embedding = x 
            x += self.embed_positions(src_tokens)
            if self.consnmt and (self.seg_embed is not None):
                x += self.seg_embed(torch.zeros_like(src_tokens))
        else:
            print("########################## THIS IS PROBABLY OKAY ##################################")
            # print('src_token',src_tokens.shape)
            # print(src_tokens)
            # print(src_tokens.shape)
            # print('src_token',src_tokens.shape)
            # print('FALSE')
            # print('src non zero',(src_tokens== decoder.sep_id).nonzero())
            sep_id_indices = (src_tokens== 4).nonzero()
            # print(sep_id_indices)
            # print(sep_id_indices.shape)
            # exit()
            sep_position = min(sep_id_indices[:,1])
            unique_sep_posi = torch.unique(sep_id_indices[:,1])
            # print('sep_posi',sep_position)
            # print('unique sep pos',unique_sep_posi)
            # exit()
            src_sent=src_tokens[:,:sep_position]
            # print('src_sent',src_sent,src_sent.shape)
            max_sep_pos = max(sep_id_indices[:,1])
            # print('max_sep_posi',max_sep_pos)
            n_sub_cons = 0
            for i in range(len(sep_id_indices)-1):
                if(sep_id_indices[i][0]==sep_id_indices[i+1][0]):
                    #print(i)
                    # print('ind',ind)
                    # print('ind_i+1', sep_id_indices[i+1])
                    n_sub_cons = sep_id_indices[i+1][1] - sep_id_indices[i][1]
                    break
            # print('n_sub_cons', n_sub_cons)
            n_cons = (max_sep_pos - sep_position)
            if(n_sub_cons > 0):
                n_cons = (n_cons // n_sub_cons) + 1
            else:
                n_cons += 1
            print('n_cons', n_cons)
            
            # print("ENCODER SIDE")
            # print(self.embed_tokens)
            # print(src_sent.shape)
            src_x = self.embed_scale * self.embed_tokens(src_sent)
            # print('src_x',src_x.shape)
            src_posi_x = self.embed_positions(src_sent) 
            src_seg_emb=self.seg_embed(torch.zeros_like(src_sent))

            cons_sent = src_tokens[:,sep_position:]
            # print('cons_sent', cons_sent)
            # print('decoder.embed_tokens(cons_sent)', decoder.embed_tokens(cons_sent))
            cons_x =  self.embed_scale * decoder.embed_tokens(cons_sent)
            # print('cons_x',cons_x.shape)
            src_embedding = torch.cat((src_x,cons_x), dim=1)
            cons_posi_x = self.cons_pos_embed(cons_sent)
            seg_cons = torch.cumsum((cons_sent==4),dim=1).type_as(cons_sent)
            seg_cons[(cons_sent==decoder.pad_id)] = torch.tensor([16]).type_as(seg_cons)  
            # print('seg_cons_shape',seg_cons[:,:-1])      
            cons_seg_emb=self.seg_embed(seg_cons)           

            x = torch.cat((src_x+src_posi_x+src_seg_emb, cons_x+cons_posi_x+cons_seg_emb),dim=1)
            # print('x_shape',x.shape)
            # print(x)
        x = self.dropout_module(x)
        # (x, p=self.dropout, training=self.training)

        # # B x T x C -> T x B x C
        # x = x.transpose(0, 1)

        # _, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        # x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            lr = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )

        #print(src_embedding)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [src_x],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            # "src_tokens": [],
            'src_tokens': [src_tokens],
            'src_embedding': [src_embedding],
            "src_lengths": [src_lengths],
            'cons_info' : {
                'sep_pos' : sep_position,
                'n_cons' : n_cons,
                'n_sub_cons' : n_sub_cons,
            },
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_embedding"]) == 0:
            src_embedding = []
        else:
            src_embedding = [(encoder_out["src_embedding"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
            "src_embedding":src_embedding,
            'cons_info' : {
                'sep_pos' : encoder_out["cons_info"]["sep_pos"],
                'n_cons' : encoder_out["cons_info"]["n_cons"],
                'n_sub_cons' : encoder_out["cons_info"]["n_sub_cons"],
            },
        }

    @torch.jit.export
    def _reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """Dummy re-order function for beamable enc-dec attention"""
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerEncoder(TransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args),
        )