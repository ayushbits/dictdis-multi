# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
    PointerNet,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerDecoderBase":
        return "TransformerDecoder"
    else:
        return module_name


class TransformerDecoderBase(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder.layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder.embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens
        self.isep_id = dictionary.isep()
        self.sep_id = dictionary.sep()
        self.pad_id = dictionary.pad()
        self.eos_id = dictionary.eos()
        self.use_ptrnet = cfg.use_ptrnet
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)

        # if self.use_ptrnet:  ## set to use pointer network
            # self.beamsize = cfg.beam if hasattr(args,'beam') else 1
            #self.beamsize = cfg.beam 
        self.ptrnet = PointerNet(cfg.encoder_embed_dim, cfg.decoder_embed_dim, len(self.dictionary))

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
            )
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.use_ptrnet:
            # ggg = torch.tensor(requires_grad=True)
            # exit()
            src_tokens = encoder_out['src_tokens'][0] if enc is not None else None 
            # print("KAAM ka HAI ye: ", src_tokens)
            # print(encoder_out.keys())
            # print("src token shape",src_tokens.shape)
            src_tokens = src_tokens.unsqueeze(1).expand(attn.size())
            #src_tokens = src_tokens[:,:,sep_tkn:]

            # print("POST SLICE src token shape",src_tokens.shape)
            # print(src_tokens.shape)
            # print("ye bhi KAAM ka HAI ye: ", src_tokens.shape)
            #exit()

            src_masks = src_tokens.eq(self.eos_id) | src_tokens.eq(self.pad_id) | src_tokens.eq(self.sep_id) | src_tokens.eq(self.isep_id)
            # print('src_masks', src_masks)
            src_embedding = encoder_out['src_embedding'][0]
            # if inference_step:
            #     a, b, c = src_embedding.shape
            #     temp = src_embedding.repeat(1,self.beamsize,1)
            #     src_embedding = temp.reshape(a*self.beamsize,b,c)
            #src_embedding = src_embedding[:,sep_tkn:,:]

            # print('src_emb',src_embedding.shape)
            # print('src_masks',src_masks.shape)

            #attn = attn[:,:,sep_tkn:]
            #print("attn shape",attn.shape)

            #souvik

            #print("sep_tkn",sep_tkn)

            dec_enc_attn = attn.masked_fill(src_masks, float(1e-15))  ##bsz x tgtlen x srclen

            #dec_enc_attn = dec_enc_attn[:,:,sep_tkn:]

            # print('dec_enc_attn',dec_enc_attn)

            enc_hids = enc.transpose(0,1) ### srclen x bsz x hidsize  -> bsz x srclen x hidsize
            #enc_hids = enc_hids[:,sep_tkn:,:]
            # print('enc_hids',enc_hids.shape)

            ctx = torch.bmm(dec_enc_attn,enc_hids)  ## bsz x tgtlen x hidsize
            # print('ctx_shape',ctx.shape)
            # print('src_embed_shape',src_embedding.transpose(1,2).shape)
            scores = torch.bmm(ctx,src_embedding.transpose(1,2))

            # src_masks = src_masks[:,:,sep_tkn:]
            #print("src mask shape",src_masks.shape)
            # print('dec enc attention', dec_enc_attn)
            # print('score ', scores)
            # print('ctx is ', ctx)
            scores = self.intra_normalization(encoder_out['cons_info'], scores)
            scores = scores.masked_fill(src_masks, float(1e-15))
            # print('scores shape' , scores.shape)
            # print('dec_enc_attn shape' , dec_enc_attn.shape)
            # print('It reached intra_normalisation')
            # print('start time is :',time.strftime("%Y-%m-%d %X"))
            
            # print('end time is :',time.strftime("%Y-%m-%d %X"))
            # print('It left intra_normalisation')
            # print('scores2',scores)
            dec_enc_attn = dec_enc_attn# + scores
            # print('scores',scores.shape)
            dec_enc_attn = self.intra_normalization(encoder_out['cons_info'], dec_enc_attn)
            # print('after normaliszaiton dec enc is ', dec_enc_attn)
            # 1/0
            
            gate = self.ptrnet(ctx, inner_states[-1].transpose(0,1))
            # print('gate ', gate)
            if torch.any(torch.isnan(scores)) or torch.any(torch.isinf(scores)):
                print('scores are ', scores)
                # print(src_tokens )
                # return None
            if torch.any(torch.isnan(gate)) or torch.any(torch.isinf(gate)):
                print('gate is ', gate)

            #src_tokens = src_tokens[:,:,sep_tkn:]
            #print("src-token shape",src_tokens.shape)
        else:
            gate, src_tokens, dec_enc_attn = None, None, None 

        # print("src_tokens",src_tokens.shape)
        # src_tokens = src_tokens[:,:,sep_tkn:]
        # print("src_tokens",src_tokens.shape)

        return x, {"attn": [attn], "inner_states": inner_states,'dec_enc_attn':dec_enc_attn, 'gate': gate, 'src_tokens': src_tokens}

## Souvik's code for intra norm
    def intra_normalization(self, cons_info, scores):
        bsz,n_tgt,src_len = scores.shape
        # print('src-mask non zero',min(src_mask.nonzero()[:,2]))
        # print('cumsum',torch.cumsum((src_mask==True),dim=2))
        # 1/0
        # sep_position = min(src_mask.nonzero()[:,:,1])
        #temp_scores = torch.zeros(bsz,n_tgt,src_len).to('cuda')
        temp_scores = torch.zeros(bsz,n_tgt,src_len)
        temp_scores = utils.move_to_cuda(temp_scores)
        
        #temp_scores = torch.zeros(bsz,n_tgt,src_len)

        n_cons = cons_info['n_cons']
        sep_pos = cons_info['sep_pos']
        n_sub_cons = cons_info['n_sub_cons']
        # print("INSIDE INTRA NORMALISATION n_cons",n_cons)
        # print("INSIDE INTRA NORMALISATION sep_pos",sep_pos)
        # print("INSIDE INTRA NORMALISATION n_sub_cons",n_sub_cons)
        temp_scores[:,:,0:sep_pos] = F.normalize(scores[:,:,0:sep_pos],p=1,dim=2)
        # x = F.normalize(scores[:,:,:],p=1,dim=2) # by ayush
        # temp_scores[:,:,0:sep_pos] = x [:,:, 0:sep_pos]
        # temp_scores[:,:,0:sep_pos] = 1e-15 # added by ayush

        if n_sub_cons:
            end_point = sep_pos + n_sub_cons
        else:
            _,_,end_point = scores.shape

        for i in range(n_cons):
            # normalise only those contraits which have a non- zero dec-enc attn (in scores)

            x = torch.count_nonzero(scores[:,:,sep_pos:end_point])
            # print("counter for normalisation",x)
            temp_scores[:,:,sep_pos:sep_pos + x ] = F.normalize(scores[:,:,sep_pos:sep_pos + x],p=1,dim=2)
            sep_pos += n_sub_cons
        scores = temp_scores

        cons_info['sep_pos'] = sep_pos - (n_sub_cons*n_cons)

        # print("SEP position debug",cons_info['sep_pos'])
        return scores
    # def intra_normalization(self, cons_info, scores):
    #     bsz,n_tgt,src_len = scores.shape
    #     # print('src-mask non zero',min(src_mask.nonzero()[:,2]))
    #     # print('cumsum',torch.cumsum((src_mask==True),dim=2))
    #     # 1/0
    #     # sep_position = min(src_mask.nonzero()[:,:,1])
    #     temp_scores = torch.zeros(bsz,n_tgt,src_len).to('cuda:0')
    #     #temp_scores = torch.zeros(bsz,n_tgt,src_len)
    #     n_cons = cons_info['n_cons']
    #     sep_pos = 0
    #     sep_pos = cons_info['sep_pos']
    #     n_sub_cons = cons_info['n_sub_cons']
    #     print('n_sub_cons ', n_sub_cons)
    #     print('sep_pos', sep_pos)
    #     print('n_cons ', n_cons)
        
    #     for i in range(n_cons):
    #         print('sep_pos ', sep_pos, ' for ',i)
    #         temp_scores[:,:,sep_pos:sep_pos + n_sub_cons] = F.normalize(scores[:,:,sep_pos:sep_pos + n_sub_cons],p=1,dim=2)
    #         sep_pos = sep_pos + n_sub_cons
        
    #     print('post sep_pos ', cons_info['sep_pos'])
    #     scores = temp_scores
    #     return scores

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        # print("NET OUTPUT", net_output[1].keys())
        logits = net_output[0].float() 
        # print("early logits",logits.shape)
        if not self.use_ptrnet:
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        
        gate = net_output[1]['gate'].float()
        dec_enc_attn = net_output[1]['dec_enc_attn'].float()
        # print('dec enc attention ', dec_enc_attn)
        src_tokens = net_output[1]['src_tokens']    
        logits = F.softmax(logits, dim=-1)

        # print(src_tokens.shape)
        sep_id = 4
        if (sep_id not in src_tokens):
            sep_position = len(src_tokens)
        else:
            sep_id_indices = (src_tokens== 4).nonzero()
            sep_position = min(sep_id_indices[:,1])  


        ######################## DEBUG ##########################

        # print("gate",gate.shape)
        # print("dec_enc_attn",dec_enc_attn.shape)
        # print("src_tokens",src_tokens.shape)
        # print("logits",logits.shape)

        # try:
        # print('gate is ', gate[0][1])
        # gate = 0.9
        # print('logits ', logits.shape)
        # sorted_log, idx = torch.sort(logits[0,0,:], descending=True)
        # print('before logits' , sorted_log[:10], idx[:10])
        


        logits = (gate * logits).scatter_add(2, src_tokens[:,:,sep_position:], (1-gate) * dec_enc_attn[:,:,sep_position:]) +1e-10
        # sorted_log, idx = torch.sort(logits[0,0,:], descending=True)
        # print('after logits' , sorted_log[:10], idx[:10])
        # print('logits are ', logits)
        # TODo
        # logits += (gate * logits).scatter_add(2, src_tokens[:,:,sep_tkn:] - 2d matrix aayega jisme fanout 1 ke indiex ho, w** dec_enc_attn[:,:,sep_tkn:]) + 1e-10
        # logits += (gate * logits).scatter_add(2, src_tokens[:,:,sep_tkn:] - 2d matrix aayega jisme fanout > 1 ke indiex ho, (1-w)* dec_enc_attn[:,:,sep_tkn:]) + 1e-10
        #logits = (gate * logits).scatter_add(2, single_src_tokens, w*(1-gate) * dec_enc_attn) + 1e-10


        # except:
        #     src_tokens = src_tokens.type(torch.int64)
        #     logits = (gate * logits).scatter_add(2, src_tokens, (1-gate) * dec_enc_attn) + 1e-10
        #     print("Kuch toh kr rha hoon")
        return torch.log(logits)      

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class TransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            TransformerConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(
            TransformerConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )
