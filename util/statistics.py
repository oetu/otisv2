# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import torch


def norm(data:torch.Tensor, attn_mask:torch.Tensor=None) -> torch.Tensor:
    """
    Zero-normalise data to have mean=0 and standard_deviation=1

    Parameters
    ----------
    data, attn_mask: (B, C, H, W)
    """
    if attn_mask is None:
        mean = torch.mean(data, dim=-1, keepdim=True)
        var = torch.var(data, dim=-1, keepdim=True)
        
        return (data - mean) / (var + 1e-9)**0.5
    else:
        # zero all entries outside the mask first, so that non-zero padding (or
        # unmasked context, e.g. a forecast-window mask) does not enter the
        # mean and variance; torch.where also ignores inf/nan outside the mask
        data = torch.where(attn_mask.bool(), data, torch.zeros_like(data))
        mean = torch.sum(data, dim=-1, keepdim=True) / (torch.sum(attn_mask, dim=-1, keepdim=True) + 1e-9)

        # aux_mask matrix is introduced to make sure that the data (zero) padding is not effected by the norm operations
        # i.e. zero entries remain untouched after the norm operations 
        aux_mask = ((1 - attn_mask) * mean)
        var = torch.sum((data - mean + aux_mask)**2, dim=-1, keepdim=True) / (torch.sum(attn_mask, dim=-1, keepdim=True) + 1e-9)
        
        return (data - mean + aux_mask) / (var + 1e-9)**0.5

def ncc(data_0:torch.Tensor, data_1:torch.Tensor, attn_mask:torch.Tensor=None, keep_batch:bool=False) -> torch.Tensor:
    """
    Zero-normalised cross-correlation coefficient between two data sets

    Zero-normalised cross-correlation equals the cosine of the angle between the unit vectors F and T, 
    being thus 1 if and only if F equals T multiplied by a positive scalar. 

    Parameters
    ----------
    data_0, data_1, attn_mask : (B, C, H, W)
    """
    nb_of_signals = 1
    if attn_mask is None:
        for dim in range(data_0.dim() - 1): # all but the last dimension (which is the actual signal)
            # (B)
            nb_of_signals = nb_of_signals * data_0.shape[dim]

        # (B, C, H)
        cross_corrs = torch.sum(norm(data=data_0) * norm(data=data_1), dim=-1) / (data_0.shape[-1] + 1e-9)
    else:
        for dim in range(data_0.dim() - 2):
            # (B)
            nb_of_signals = nb_of_signals * data_0.shape[dim]

        # check whether a channel is padding (only zero elements) or signal (at least one non-zero element)
        # (B, C)
        nb_of_channels = torch.max(attn_mask, dim=-1)[0].sum(-1)
        nb_of_signals = nb_of_signals * nb_of_channels
        # (B, C, H)
        nb_of_signals = nb_of_signals.unsqueeze(-1)

        # (B, C, H)
        cross_corrs = torch.sum(norm(data=data_0, attn_mask=attn_mask) * norm(data=data_1, attn_mask=attn_mask), dim=-1) / (torch.sum(attn_mask, dim=-1) + 1e-9)

    if keep_batch == True:
        # compute ncc of each sample
        nb_of_signals = nb_of_signals / cross_corrs.shape[0]
        # (B, C, H)
        ncc = cross_corrs / nb_of_signals
        # (B)
        ncc = ncc.flatten(1).sum(-1)
    else:
        # compute ncc of the entire batch
        ncc = (cross_corrs / nb_of_signals).sum()

    return ncc

def masked_mean(tensor, attn_mask=None, dim=None, keep_dim=False):
    """
        Determine the mean while considering the attention mask

        tensor: (B, N, D)
        attn_mask: (B, N)
    """
    if attn_mask is None:
        if dim is None:
            return torch.mean(tensor)
        else:
            return torch.mean(tensor, dim=dim, keepdim=keep_dim)
    else:
        tensor_masked = tensor * attn_mask.unsqueeze(-1)
        if dim is None:
            return torch.sum(tensor_masked) / (torch.sum(attn_mask.unsqueeze(-1).repeat(1, 1, tensor.shape[-1])) + 1e-9)
        else:
            numerator = torch.sum(tensor_masked, dim=dim, keepdim=keep_dim)
            denominator = torch.sum(attn_mask.unsqueeze(-1).repeat(1, 1, tensor.shape[-1]), dim=dim, keepdim=keep_dim) + 1e-9
            return numerator / denominator

def koleo_loss(tokens: torch.Tensor, group_size: int = None, eps: float = 1e-8) -> torch.Tensor:
    """
    KoLeo regulariser (Kozachenko-Leonenko estimator).
    Encourages a uniform spread of tokens in the embedding space by maximising
    the average log-distance to each token's nearest neighbour.

    Returns 0 when fewer than 2 tokens are provided.

    Reference: Sablayrolles et al., "Spreading vectors for similarity search", ICLR 2019
               (used in DINOv2 / DINOv3 for [CLS] / patch token regularisation)

    Parameters
    ----------
    tokens : (K, D) — token embeddings
    group_size : int, optional
        If set, split the batch into non-overlapping groups of this many tokens
        and compute KoLeo per group, then average. Matches DINOv3's
        "distributed implementation of Koleo ... applied in small batches of 16
        samples" (DINOv3 §3.2). The tail (K % group_size) is dropped, and
        the loss is 0 when K < group_size (e.g. a small last batch), so the
        estimator always uses groups of exactly group_size tokens.

    Returns
    -------
    scalar loss (lower = more uniform)
    """
    if tokens.shape[0] < 2:
        return torch.tensor(0.0, device=tokens.device)

    # compute in float32 — latent comes from autocast (float16) and the
    # pairwise-distance formula (2 - 2*dot) suffers from catastrophic
    # cancellation in half precision when tokens are similar
    tokens = tokens.float()
    tokens = torch.nn.functional.normalize(tokens, dim=-1)

    if group_size is not None:
        if tokens.shape[0] < group_size:
            # too few tokens for one full group: no KoLeo for this batch
            return torch.tensor(0.0, device=tokens.device)
        # drop tail so every group has exactly group_size tokens
        n_groups = tokens.shape[0] // group_size
        tokens = tokens[: n_groups * group_size].reshape(n_groups, group_size, -1)
        # pairwise distances per group  (G, g, g)
        dots = torch.bmm(tokens, tokens.transpose(1, 2))
        dists = torch.sqrt((2.0 - 2.0 * dots).clamp(min=1e-8))
        inf_diag = torch.diag(torch.full((group_size,), float('inf'), device=tokens.device))
        dists = dists + inf_diag.unsqueeze(0)
        min_dists, _ = dists.min(dim=-1)
        return -torch.log(min_dists + eps).mean()

    # pairwise distances  (K, K)
    dots = torch.mm(tokens, tokens.T)
    dists = torch.sqrt((2.0 - 2.0 * dots).clamp(min=1e-8))
    # mask out self-distances without inplace op (fill_diagonal_ breaks autograd)
    inf_diag = torch.diag(torch.full((tokens.shape[0],), float('inf'), device=tokens.device))
    dists = dists + inf_diag
    # nearest-neighbour distance for each token
    min_dists, _ = dists.min(dim=-1)
    # KoLeo loss = negative mean log nearest-neighbour distance
    return -torch.log(min_dists + eps).mean()

def rankme(z: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    RankMe — effective-rank diagnostic of self-supervised representations.

    Computes the exponentiated Shannon entropy of the singular-value spectrum:
        RankMe(Z) = exp(-Σ_i p_i log p_i),    p_i = σ_i(Z) / ‖σ(Z)‖_1 + ε
    Bounded in [1, min(n, d)]. Higher = more directions used = less collapsed.

    ε is added AFTER the ratio, as the reference writes it.

    NB on Z. The reference defines Z as N SAMPLE-LEVEL embeddings × d ("we use
    25600 samples"). This function takes whatever Z the caller builds, and the
    repo builds three:

      * ``train_rankme``     — PATCH TOKENS, i.e. within-sample spatial
                               diversity rather than the across-sample quantity
                               the paper defines. A deviation; read on its own
                               scale.
      * ``train_rankme_gap`` — mean over a sample's real visible patch tokens.
      * ``train_rankme_cls`` — the sample's [CLS] token.

    The last two are the reference's Z, but online they are bounded by
    min(micro-batch, D), so they are censored whenever the micro-batch is
    smaller than the encoder width and are NOT comparable to an offline
    protocol with N = 25,600 samples.

    Reference: Garrido, Balestriero, Najman, LeCun.
        "RankMe: Assessing the Downstream Performance of Pretrained
         Self-Supervised Representations by Their Rank", ICML 2023.

    Parameters
    ----------
    z : (..., D) tensor of token embeddings (raw, un-normalised).
        Higher-rank inputs are flattened to (N, D); D is the last dim.
    eps : numerical stabiliser added to each normalised singular value.

    Returns
    -------
    scalar tensor on z.device.
    """
    if z.dim() > 2:
        z = z.reshape(-1, z.shape[-1])
    # eigvalsh of the smaller-side Gram matrix avoids a full SVD; cuSOLVER prefers
    # fp32 to bf16/fp16, so we cast. Wrapped in no_grad: this is a diagnostic and
    # backward through eigvalsh is unstable near degenerate spectra.
    with torch.no_grad():
        z = z.float()
        n, d = z.shape
        if n >= d:
            gram = z.T @ z                                # (d, d)
        else:
            gram = z @ z.T                                # (n, n)
        evals = torch.linalg.eigvalsh(gram).clamp(min=0)
        singular = evals.sqrt()
        total = singular.sum()
        if total <= 0:
            return torch.ones((), device=z.device)
        p = singular / total + eps
        entropy = -(p * p.log()).sum()
        return entropy.exp()

def masked_var(tensor, attn_mask=None, dim=None, keep_dim=False):
    """
        Determine the variance while considering the attention mask

        tensor: (B, N, D)
        attn_mask: (B, N)
    """
    if attn_mask is None:
        if dim is None:
            return torch.var(tensor)
        else:
            return torch.var(tensor, dim=dim, keepdim=keep_dim)
    else:
        tensor_masked = tensor * attn_mask.unsqueeze(-1)
        deviations = tensor_masked - masked_mean(tensor, attn_mask, dim=dim, keep_dim=True)
        deviations = deviations * attn_mask.unsqueeze(-1)
        squared_deviations = (deviations ** 2) * attn_mask.unsqueeze(-1)
        if dim is None:
            return torch.sum(squared_deviations) / (torch.sum(attn_mask.unsqueeze(-1).repeat(1, 1, tensor.shape[-1])) - 1 + 1e-9)
        else:
            return torch.sum(squared_deviations, dim=dim, keepdim=keep_dim) / (torch.sum(attn_mask.unsqueeze(-1).repeat(1, 1, tensor.shape[-1]), dim=dim, keepdim=keep_dim) - 1 + 1e-9)