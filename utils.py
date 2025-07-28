# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for GPTrajectory.

References:
    https://github.com/karpathy/minGPT
"""
import numpy as np
import os
import logging
import random
import datetime
import socket


import torch
import torch.nn as nn
from torch.nn import functional as F

torch.pi = torch.acos(torch.zeros(1)).item() * 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def new_log(logdir, filename):
    """Defines logging format."""
    filename = os.path.join(
        logdir,
        datetime.datetime.now().strftime(
            "log_%Y-%m-%d-%H-%M-%S_" + socket.gethostname() + "_" + filename + ".log"
        ),
    )
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        format="%(asctime)s - %(name)s - %(message)s",
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def haversine(input_coords, pred_coords):
    """Calculate the haversine distances between input_coords and pred_coords.

    Args:
        input_coords, pred_coords: Tensors of size (...,N), with (...,0) and (...,1) are
        the latitude and longitude in radians.

    Returns:
        The havesine distances between
    """
    R = 6371
    lat_errors = pred_coords[..., 0] - input_coords[..., 0]
    lon_errors = pred_coords[..., 1] - input_coords[..., 1]
    a = (
        torch.sin(lat_errors / 2) ** 2
        + torch.cos(input_coords[:, :, 0])
        * torch.cos(pred_coords[:, :, 0])
        * torch.sin(lon_errors / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    d = R * c
    return d


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def top_k_nearest_idx(att_logits, att_idxs, r_vicinity):
    """Keep only k values nearest the current idx.

    Args:
        att_logits: a Tensor of shape (bachsize, data_size).
        att_idxs: a Tensor of shape (bachsize, 1), indicates
            the current idxs.
        r_vicinity: number of values to be kept.
    """
    device = att_logits.device
    idx_range = (
        torch.arange(att_logits.shape[-1]).to(device).repeat(att_logits.shape[0], 1)
    )
    idx_dists = torch.abs(idx_range - att_idxs)
    out = att_logits.clone()
    out[idx_dists >= r_vicinity / 2] = -float('Inf')
    return out


@torch.no_grad()
def sample(
    model,
    seqs,
    steps,
    temperature=1.0,
    sample=False,
    sample_mode="pos_vicinity",
    r_vicinity=20,
    top_k=None,
):
    """
    Take a conditoning sequence of AIS observations seq and predict the next observation,
    feed the predictions back into the model each time.
    """
    max_seqlen = model.get_max_seqlen()
    model.eval()
    for k in range(steps):
        seqs_cond = (
            seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]
        )  # crop context if needed

        # logits.shape: (batch_size, seq_len, data_size)
        logits, _ = model(seqs_cond)
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seqs.device) + 0.5

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature  # (batch_size, data_size)

        lat_logits, lon_logits, sog_logits, cog_logits = torch.split(
            logits,
            (model.lat_size, model.lon_size, model.sog_size, model.cog_size),
            dim=-1,
        )

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = top_k_logits(lat_logits, top_k)
            lon_logits = top_k_logits(lon_logits, top_k)
            sog_logits = top_k_logits(sog_logits, top_k)
            cog_logits = top_k_logits(cog_logits, top_k)

        # apply softmax to convert to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)

        # sample from the distribution or take the most likely
        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)  # (batch_size, 1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        # convert to x (range: [0,1))
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes

        # append to the sequence and continue
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)

    return seqs
