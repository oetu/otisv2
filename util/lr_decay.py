# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:     https://github.com/facebookresearch/mae
# ELECTRA: https://github.com/google-research/electra
# --------------------------------------------------------

import json


def _is_muon_param(name, param):
    """Determine if a parameter should use the MUON optimizer."""
    if param.ndim < 2:
        return False
    if name.endswith(".bias"):
        return False
    # Learnable tokens / queries (any *_token(s), e.g. mask_token_encoder, and
    # the attention-pool query q) are not weight matrices: use AdamW
    leaf = name.split(".")[-1]
    if leaf.endswith("_token") or leaf.endswith("_tokens") or leaf.startswith("mask_token") or name == "q":
        return False
    # DINO / iBOT prototype layers (``*.last_layer.*``): AdamW, as in DINOv2/v3
    if ".last_layer." in name:
        return False
    if "norm" in name:
        return False
    if name.startswith("decoder_pred") or name.startswith("mlp_pred") or name.startswith("head"):
        return False
    return True


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75,
                     use_muon=False):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        # Determine MUON eligibility
        muon_flag = use_muon and _is_muon_param(n, p)
        muon_suffix = "_muon" if muon_flag else "_adam"

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s%s" % (layer_id, g_decay, muon_suffix)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            group_dict = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            if use_muon:
                group_dict["use_muon"] = muon_flag

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            if use_muon:
                param_group_names[group_name]["use_muon"] = muon_flag
            param_groups[group_name] = group_dict

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'register_tokens']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers