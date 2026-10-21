"""Loss functions for metric learnings.

References
----------
- https://kevinmusgrave.github.io/pytorch-metric-learning/losses/

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""


import typing

import torch
import typing_extensions

import torch_utils.distributed as dist
import torch_utils.training_stats as training_stats
import torch_utils.util as util

# ----------------------------------------------------------------------------------------------------------------------------------
# Utility functions


def compute_l2_distance(tensor_a: torch.Tensor, tensor_b: torch.Tensor, eps: float) -> torch.Tensor:
    diff = tensor_a - tensor_b
    return ((diff * diff).sum(dim=-1) + eps).sqrt()


def _pairwise_l2_distance(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute pairwise L2 distance matrix for a batch of vectors. This function internally uses GEMM for efficient computation. Surprisingly, faster and more memory efficient than `torch.cdist` for large batch sizes.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (N, D).
    eps : float
        A small value added for numerical stability to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Pairwise L2 distance matrix of shape (N, N).
    """

    x2 = (x * x).sum(dim=1, keepdim=True)  # (N, 1)
    dist2 = x2 + x2.T - 2.0 * (x @ x.T)    # (N, N)
    dist2 = dist2.clamp_min_(0.0)

    return (dist2 + eps).sqrt()

# ----------------------------------------------------------------------------------------------------------------------------------
# Base Class


class MetricLossBase(torch.nn.Module):
    """Base class for metric learning loss functions."""

    def __init__(self):
        super(MetricLossBase, self).__init__()

    def forward(self, net: torch.nn.Module, imgs: torch.Tensor, cls_labels: torch.Tensor, params: torch.Tensor | None) -> torch.Tensor:
        """Compute the loss.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network to compute embeddings.
        imgs : torch.Tensor
            Input images, shape (B, C, H, W).
        cls_labels : torch.Tensor
            Discrete class labels corresponding to the input images, shape (B,).
        params : torch.Tensor | None
            Parameters corresponding to the input images, shape (B, D).

        Returns
        -------
        torch.Tensor
            The computed loss.
        """

        raise NotImplementedError('MetricLossBase is an abstract class and cannot be used directly.')


# ----------------------------------------------------------------------------------------------------------------------------------
# Siamese Loss (Contrastive Loss without 1/2 scaling factor)


class SiameseLoss(MetricLossBase):
    """Siamese loss for metric learning. This equals to the contrastive loss without the scaling factor of 1/2.

    Reference
    ----------

    The implementation of contrastive Loss in `pytorch-metric-learning` library.

    You can find the way of computing the loss in the following link:
        https://github.com/KevinMusgrave/pytorch-metric-learning/blob/c8350998ebc8aacf2c45de50e2556bc854cc0361/src/pytorch_metric_learning/losses/contrastive_loss.py

    You can find the way of selecting positive and negative pairs in the following link:
        https://github.com/KevinMusgrave/pytorch-metric-learning/blob/c8350998ebc8aacf2c45de50e2556bc854cc0361/src/pytorch_metric_learning/losses/generic_pair_loss.py#L17
    """

    def __init__(self, margin: float = 1.0, norm_emb: bool = True, eps: float = 1e-8, unwrap_dist: bool = False, **kwargs):
        """Initialize the SiameseLoss.

        Parameters
        ----------
        margin : float
            Margin value for the loss. This value works as a threshold for negative pairs because this value is defined as a absolute value.
        norm_emb : bool
            If True, normalize embeddings before computing distances.
        eps : float
            A small value added for numerical stability to avoid division by zero.
        unwrap_dist : bool
            If True, unwraps the distance by calculating the power of `p` after computing the pairwise differences.
        """

        super(SiameseLoss, self).__init__()

        self._margin = margin
        self._norm_emb = norm_emb
        self._eps = eps
        self._unwrap_dist = unwrap_dist

    @typing_extensions.override
    def forward(self, net: torch.nn.Module, imgs: torch.Tensor, cls_labels: torch.Tensor, params: torch.Tensor | None = None) -> torch.Tensor:
        assert cls_labels.ndim == 1, f"cls_labels.ndim must be 1, but got {cls_labels.ndim}"  # (N,)
        # NOTE: `params` is unused here. It is accepted (and ignored) so that the training loop can feed the exact same
        # batch dictionary to every loss in the catalog, which keeps the data pipeline identical across runs.

        # ----------------------------------------------------------------------------------------------------
        # Forward pass

        embs = net(imgs)  # (N, D)

        assert embs.ndim == 2, f"embs.ndim must be 2, but got {embs.ndim}"  # (N, D)

        # ----------------------------------------------------------------------------------------------------
        # Compute pairwise distances

        if self._norm_emb:
            embs = torch.nn.functional.normalize(embs, p=2, dim=-1)

        # Distance computation
        #   - see also: https://github.com/KevinMusgrave/pytorch-metric-learning/blob/c8350998ebc8aacf2c45de50e2556bc854cc0361/src/pytorch_metric_learning/distances/base_distance.py#L39
        #   - According to "https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#contrastiveloss", the instance of `LpDistance(normalize_embeddings=True, p=2, power=1)` is used for distance computation.
        #   Therefore, we can use the `torch.cdist` function directly here and don't need to compute squared distances.

        dist = ((embs.unsqueeze(1) - embs.unsqueeze(0)).pow(2).sum(dim=-1) + self._eps)  # (B, B)
        if not self._unwrap_dist:
            dist = dist.pow(0.5)  # Compute L2 distance if not unwrapping
        assert dist.isfinite().all(), "Non-finite values found in distance matrix."
        assert dist.shape == (embs.shape[0], embs.shape[0]), f"dist.shape must be {(embs.shape[0], embs.shape[0])}, but got {dist.shape}"

        # ----------------------------------------------------------------------------------------------------
        # Compute loss over pairs

        pair_mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)  # (B, B), bool
        labels_equal = cls_labels.unsqueeze(0) == cls_labels.unsqueeze(1)                    # (B, B), bool

        pos_mask = pair_mask & labels_equal        # (B, B), bool
        neg_mask = pair_mask & (~labels_equal)     # (B, B), bool

        pos_dists = dist[pos_mask]  # (num_pos_pairs,)
        pos_loss = pos_dists.sum()

        neg_dists = dist[neg_mask]  # (num_neg_pairs,)
        if self._unwrap_dist:
            neg_loss = torch.clamp(self._margin - neg_dists.pow(0.5), min=0.0).pow(2).sum()
        else:
            neg_loss = torch.clamp(self._margin - neg_dists, min=0.0).sum()

        num_pos = pos_mask.sum()
        num_neg = neg_mask.sum()
        num_pairs = (num_pos + num_neg).clamp(min=1.0).float()

        loss = (pos_loss + neg_loss) / num_pairs

        return loss

# ----------------------------------------------------------------------------------------------------------------------------------
# Triplet Loss


class TripletMarginLoss(MetricLossBase):
    """Triplet margin loss for metric learning. The positive and negative samples are selected based on the discrete class labels.

    This class is intentionally kept structurally identical to `ParameterAwareTripletMarginLoss` so that the two losses
    differ *only* in how positives and negatives are assigned: this class uses the discrete class labels (all samples
    sharing the anchor's class are positives, all others are negatives), whereas `ParameterAwareTripletMarginLoss`
    ranks two candidates by their distance in the continuous parameter space. Everything else -- the distance routine
    (`_pairwise_l2_distance`), the reduction and the reported statistics -- is shared, so that the two can be compared
    under identical conditions.

    NOTE: The reduction differs from the default of `pytorch-metric-learning`, whose `TripletMarginLoss` reduces with
    `AvgNonZeroReducer` (i.e., averages over the non-zero terms only). Here the sum is divided by the total number of
    mined triplets, which is the reduction used by `ParameterAwareTripletMarginLoss`.

    See also:
        - https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tripletmarginloss (for the reference implementation)
        - https://github.com/KevinMusgrave/pytorch-metric-learning/blob/c8350998ebc8aacf2c45de50e2556bc854cc0361/src/pytorch_metric_learning/distances/lp_distance.py#L7 (for distance computation)
    """

    def __init__(self, margin: float = 0.05, norm_emb: bool = True, eps: float = 1e-8, log_stats: bool = True, **kwargs):
        """Initialize the TripletMarginLoss.

        Parameters
        ----------
        margin : float
            Margin value for the triplet loss. This value is added to the distance between anchor and negative samples. We can regard this value as a relative pushing power for negative samples compared to positive samples.
        norm_emb : bool
            If True, normalize embeddings before computing distances.
        eps : float
            A small value to avoid numerical instability.
        log_stats : bool
            If True, log statistics about the loss.
        """

        super(TripletMarginLoss, self).__init__()

        self._margin = margin
        self._norm_emb = norm_emb
        self._eps = eps
        self._log_stats = log_stats

        dist.print0(f'TripletMarginLoss initialized with margin={margin}, norm_emb={norm_emb}, eps={eps}, log_stats={log_stats}\n')

    def _compute_triplet_distance(self, embs: torch.Tensor, cls_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ----------------------------------------------------------------------------------------------------
        # Find triplets
        #   see also: https://github.com/KevinMusgrave/pytorch-metric-learning/blob/c8350998ebc8aacf2c45de50e2556bc854cc0361/src/pytorch_metric_learning/losses/triplet_margin_loss.py

        matches = cls_labels.unsqueeze(1) == cls_labels.unsqueeze(0)  # Find positives
        diffs = matches.logical_not()
        matches.fill_diagonal_(False)

        triplets_bool = matches.unsqueeze(2) & diffs.unsqueeze(1)  # Find negatives, therefore triplets
        anchor_idx, positive_idx, negative_idx = torch.where(triplets_bool)

        num_triplets = anchor_idx.shape[0]
        assert num_triplets > 0, "No valid triplet found. Ensure that the batch contains at least one class with at least 2 samples and at least one sample from a different class."

        # ----------------------------------------------------------------------------------------------------
        # Compute distances of embeddings
        #   NOTE: The distances are gathered from the (N, N) pairwise matrix instead of gathering the embeddings
        #   themselves, because the number of triplets grows as O(N^3) and materializing (num_triplets, D) tensors
        #   would blow up the memory footprint. This also keeps the distance computation identical to
        #   `ParameterAwareTripletMarginLoss`.

        if self._norm_emb:
            embs = torch.nn.functional.normalize(embs, p=2.0, dim=-1)

        emb_dist_mat = _pairwise_l2_distance(embs, eps=self._eps)  # (N, N)

        ap_dist = emb_dist_mat[anchor_idx, positive_idx]  # (num_triplets,)
        an_dist = emb_dist_mat[anchor_idx, negative_idx]  # (num_triplets,)

        return anchor_idx, positive_idx, negative_idx, ap_dist, an_dist

    def forward_impl(self, embs: torch.Tensor, cls_labels: torch.Tensor) -> torch.Tensor:
        """Forward pass implementation of Triplet Margin Loss.

        Parameters
        ----------
        embs : torch.Tensor
            Embeddings of shape `(N, D)`.
        cls_labels : torch.Tensor
            Discrete class labels of shape `(N,)`.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """

        assert cls_labels.ndim == 1, f"cls_labels.ndim must be 1, but got {cls_labels.ndim}"  # (N,)

        # ----------------------------------------------------------------------------------------------------
        # Compute triplet distances

        anchor_idx, _, _, ap_dist, an_dist = self._compute_triplet_distance(embs=embs, cls_labels=cls_labels)

        num_triplets = anchor_idx.shape[0]

        # ----------------------------------------------------------------------------------------------------
        # Compute triplet margin loss: max(0, d(a,p) - d(a,n) + margin)

        violation = ap_dist - an_dist + self._margin   # (num_triplets,)
        loss = torch.nn.functional.relu(violation)     # (num_triplets,)

        assert loss.ndim == 1, f"loss.ndim must be 1, but got {loss.ndim}"  # (num_triplets,)

        loss = loss.sum().div(num_triplets)  # scalar

        # ----------------------------------------------------------------------------------------------------
        # Log statistics

        if self._log_stats:
            with torch.no_grad():
                delta = an_dist - ap_dist  # (num_triplets,)

                triplet_ratio_hard = (delta <= 0.0).float().mean()
                triplet_ratio_semi_hard = ((delta > 0.0) & (delta <= self._margin)).float().mean()
                triplet_ratio_easy = (delta > self._margin).float().mean()
                triplet_nonzero_ratio = (violation > 0.0).float().mean()

                ap_dist_mean = ap_dist.mean()
                ap_dist_std = ap_dist.std(unbiased=False)
                an_dist_mean = an_dist.mean()
                an_dist_std = an_dist.std(unbiased=False)
                dist_gap_mean = (an_dist - ap_dist).mean()

                num_unique_anchors = torch.unique(anchor_idx).numel()

                # autopep8: off
                training_stats.report('metric_stats/num_triplets',       num_triplets)
                training_stats.report('metric_stats/num_unique_anchors', num_unique_anchors)
                training_stats.report('metric_stats/hard_ratio',         triplet_ratio_hard)
                training_stats.report('metric_stats/semi_hard_ratio',    triplet_ratio_semi_hard)
                training_stats.report('metric_stats/easy_ratio',         triplet_ratio_easy)
                training_stats.report('metric_stats/nonzero_ratio',      triplet_nonzero_ratio)
                training_stats.report('metric_stats/ap_dist_mean',       ap_dist_mean)
                training_stats.report('metric_stats/ap_dist_std',        ap_dist_std)
                training_stats.report('metric_stats/an_dist_mean',       an_dist_mean)
                training_stats.report('metric_stats/an_dist_std',        an_dist_std)
                training_stats.report('metric_stats/dist_gap_mean',      dist_gap_mean)
                # autopep8: on

        return loss

    @typing_extensions.override
    def forward(self, net: torch.nn.Module, imgs: torch.Tensor, cls_labels: torch.Tensor, params: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of Triplet Margin Loss.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network to compute embeddings.
        imgs : torch.Tensor
            Input images.
        cls_labels : torch.Tensor
            Discrete class labels used to select positive and negative samples.
        params : torch.Tensor | None
            Unused. Accepted so that this loss is a drop-in replacement for `ParameterAwareTripletMarginLoss`, which lets
            the training loop feed the exact same batch dictionary to either loss.

        Returns
        -------
        torch.Tensor
            Loss value.
        """

        # ----------------------------------------------------------------------------------------------------
        # Forward pass

        embs: torch.Tensor = net(imgs)  # (N, D)
        assert embs.ndim == 2, f"embs.ndim must be 2, but got {embs.ndim}"  # (N, D)

        # ----------------------------------------------------------------------------------------------------
        # Compute loss

        loss = self.forward_impl(embs=embs, cls_labels=cls_labels)

        return loss

# ----------------------------------------------------------------------------------------------------------------------------------
# Parameter Aware Triplet Margin Loss


class MinerType(util.StrEnum):
    """Type of miner to select triplets."""

    all = 'all'                        # Use all possible triplets within the batch
    each_class = 'each_class'          # Select one pair from the same class and another from different classes
    same_class = 'same_class'          # Select all pairs from the same class
    diff_class = 'diff_class'          # Select all pairs from different classes


class ParameterAwareTripletMarginLoss(MetricLossBase):
    """Parameter Aware Triplet Margin Loss for metric learning. The positive and negative samples are selected based on the distances in the label space."""

    _all_pairs_cache: dict[tuple[int, str, int | None], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __init__(self, miner: MinerType, margin: float = 0.05, norm_emb: bool = True, eps: float = 1e-8, log_stats: bool = True, **kwargs):
        """Initialize the ParameterAwareTripletMarginLoss.

        Parameters
        ----------
        miner : MinerType
            Type of miner to select triplets.
        margin : float
            Margin value for the triplet loss. This value is added to the distance between anchor and negative samples.
        norm_emb : bool
            If True, normalize embeddings before computing distances.
        eps : float
            A small value to avoid numerical instability.
        log_stats : bool
            If True, log statistics about the loss.
        """

        super(ParameterAwareTripletMarginLoss, self).__init__()

        self._miner = miner
        self._margin = margin
        self._norm_emb = norm_emb
        self._eps = eps
        self._log_stats = log_stats

        dist.print0(f'ParameterAwareTripletMarginLoss initialized with miner={miner}, margin={margin}, norm_emb={norm_emb}, eps={eps}, log_stats={log_stats}\n')

    @staticmethod
    def compute_all_pairs_excluding_self(N: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all triplets (anchor, pair0, pair1) where pair0 != anchor, pair1 != anchor, and pair0 < pair1 in the anchor-excluded list.

        Parameters
        ----------
        N : int
            Number of samples in the batch.
        device : torch.device
            Device on which to create the tensors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            anchor_idx, pair_idx_0, pair_idx_1
            Each has shape (N * C(N - 1, 2),).
        """

        assert N >= 3, f"N must be at least 3, but got {N}"

        axial_idx = torch.arange(N, device=device)  # (N,)

        # others[a] = [0, 1, ..., a-1, a+1, ..., N-1]
        mask = ~torch.eye(N, dtype=torch.bool, device=device)   # (N, N)
        others = axial_idx.expand(N, N)[mask].reshape(N, N - 1)  # (N, N-1)

        O = N - 1

        # combinations over positions within each row of others
        comb = torch.combinations(torch.arange(O, device=device), r=2)  # (C_val, 2)
        comb_i = comb[:, 0]  # (C_val,)
        comb_j = comb[:, 1]  # (C_val,)
        C_val = comb.shape[0]

        # anchor ids repeated for each local combination
        anchor_idx = axial_idx.repeat_interleave(C_val)  # (N * C_val,)

        # For each output row, which anchor row in `others` should be referenced?
        anchor_row = torch.arange(N, device=device).repeat_interleave(C_val)  # (N * C_val,)

        # Repeat combination indices across anchors
        comb_i_rep = comb_i.repeat(N)  # (N * C_val,)
        comb_j_rep = comb_j.repeat(N)  # (N * C_val,)

        pair_idx_0 = others[anchor_row, comb_i_rep]  # (N * C_val,)
        pair_idx_1 = others[anchor_row, comb_j_rep]  # (N * C_val,)

        assert anchor_idx.shape == pair_idx_0.shape == pair_idx_1.shape

        return anchor_idx, pair_idx_0, pair_idx_1

    @classmethod
    def get_all_pairs_excluding_self_cached(cls, N: int, device: torch.device,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all triplets (anchor, pair0, pair1) where pair0 != anchor, pair1 != anchor, and pair0 < pair1 in the anchor-excluded list, with caching.

        Parameters
        ----------
        N : int
            Number of samples in the batch.
        device : torch.device
            Device on which to create the tensors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            anchor_idx, pair_idx_0, pair_idx_1
            Each has shape (N * C(N - 1, 2),).
        """

        key = (N, device.type, device.index)

        if key not in cls._all_pairs_cache:
            cls._all_pairs_cache[key] = cls.compute_all_pairs_excluding_self(N=N, device=device)

        return cls._all_pairs_cache[key]

    def _compute_triplet_distance(self, embs: torch.Tensor, cls_labels: torch.Tensor, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = params.shape[0]
        device = params.device

        # ----------------------------------------------------------------------------------------------------
        # Find triplets:  anchor i, and two random "others"

        if self._miner == MinerType.all:
            anchor_idx, pair_idx_0, pair_idx_1 = self.get_all_pairs_excluding_self_cached(N=batch_size, device=device)
        elif self._miner == MinerType.each_class:
            # For each anchor, select one pair from the same class and another from different classes

            anchor_idx_all, pair_idx_0_all, pair_idx_1_all = self.get_all_pairs_excluding_self_cached(N=batch_size, device=device)

            cls_anchor = cls_labels[anchor_idx_all]    # (batch_size * C(batch_size - 1, 2),)
            cls_pair_0 = cls_labels[pair_idx_0_all]    # (batch_size * C(batch_size - 1, 2),)
            cls_pair_1 = cls_labels[pair_idx_1_all]    # (batch_size * C(batch_size - 1, 2),)
            same_0 = (cls_anchor == cls_pair_0)        # (batch_size * C(batch_size - 1, 2),)
            same_1 = (cls_anchor == cls_pair_1)        # (batch_size * C(batch_size - 1, 2),)

            valid = same_0 ^ same_1                    # XOR condition: (same0 or not same1) or (not same0 or same1)

            # Sanity check
            if not valid.any():
                raise ValueError('No valid triplet found for MinerType.each. Ensure that the batch contains at least 2 different classes with at least 2 samples each.')

            anchor_idx = anchor_idx_all[valid]        # (num_triplets,)
            pair_idx_0 = pair_idx_0_all[valid]        # (num_triplets,)
            pair_idx_1 = pair_idx_1_all[valid]        # (num_triplets,)
        elif self._miner == MinerType.same_class:
            # Select 3 pairs from the same class based on discrete class labels and then use label distances to determine positive/negative

            anchor_all, pair_0_all, pair_1_all = self.get_all_pairs_excluding_self_cached(N=batch_size, device=device)  # (batch_size * C(batch_size - 1, 2),)

            cls_anchor = cls_labels[anchor_all]    # (batch_size * C(batch_size - 1, 2),)
            cls_pair_0 = cls_labels[pair_0_all]    # (batch_size * C(batch_size - 1, 2),)
            cls_pair_1 = cls_labels[pair_1_all]    # (batch_size * C(batch_size - 1, 2),)

            valid = (cls_anchor == cls_pair_0) & (cls_anchor == cls_pair_1)  # (batch_size * C(batch_size - 1, 2),)
            if not valid.any():
                raise ValueError('No valid triplet found for MinerType.same_class. Ensure that the batch contains at least one class with at least 3 samples.')

            anchor_idx = anchor_all[valid]        # (num_triplets,)
            pair_idx_0 = pair_0_all[valid]        # (num_triplets,)
            pair_idx_1 = pair_1_all[valid]        # (num_triplets,)
        elif self._miner == MinerType.diff_class:
            # Select 3 pairs from different classes based on discrete class labels and then use label distances to determine positive/negative

            anchor_all, pair_0_all, pair_1_all = self.get_all_pairs_excluding_self_cached(N=batch_size, device=device)  # (batch_size * C(batch_size - 1, 2),)

            cls_anchor = cls_labels[anchor_all]    # (batch_size * C(batch_size - 1, 2),)
            cls_pair_0 = cls_labels[pair_0_all]    # (batch_size * C(batch_size - 1, 2),)
            cls_pair_1 = cls_labels[pair_1_all]    # (batch_size * C(batch_size - 1, 2),)

            valid = (cls_anchor != cls_pair_0) & (cls_anchor != cls_pair_1) & (cls_pair_0 != cls_pair_1)  # (batch_size * C(batch_size - 1, 2),)
            if not valid.any():
                raise ValueError('No valid triplet found for MinerType.diff_class. Ensure that the batch contains at least 3 different classes.')

            anchor_idx = anchor_all[valid]        # (num_triplets,)
            pair_idx_0 = pair_0_all[valid]        # (num_triplets,)
            pair_idx_1 = pair_1_all[valid]        # (num_triplets,)
        else:
            raise ValueError(f'Unknown miner type: {self._miner}')

        num_triplets = anchor_idx.shape[0]  # == N * n_pairs
        assert num_triplets > 0, "No valid triplet found. Ensure that the batch contains enough samples to form triplets based on the selected miner type."

        # ----------------------------------------------------------------------------------------------------
        # Determine positive/negative based on continuous label distances

        param_dist_mat = _pairwise_l2_distance(params, eps=self._eps)  # (N, N)

        pair_param_dist_0 = param_dist_mat[anchor_idx, pair_idx_0]  # (num_triplets,)
        pair_param_dist_1 = param_dist_mat[anchor_idx, pair_idx_1]  # (num_triplets,)

        mask = pair_param_dist_0 <= pair_param_dist_1

        positive_idx = torch.where(mask, pair_idx_0, pair_idx_1)
        negative_idx = torch.where(mask, pair_idx_1, pair_idx_0)

        # ----------------------------------------------------------------------------------------------------
        # Compute distances of embeddings

        if self._norm_emb:
            embs = torch.nn.functional.normalize(embs, p=2.0, dim=-1)

        emb_dist_mat = _pairwise_l2_distance(embs, eps=self._eps)  # (N, N)

        ap_dist = emb_dist_mat[anchor_idx, positive_idx]  # (num_triplets,)
        an_dist = emb_dist_mat[anchor_idx, negative_idx]  # (num_triplets,)

        return anchor_idx, positive_idx, negative_idx, ap_dist, an_dist, pair_param_dist_0, pair_param_dist_1

    def forward_impl(self, embs: torch.Tensor, cls_labels: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Forward pass implementation of Parameter Aware Triplet Margin Loss.

        Parameters
        ----------
        embs : torch.Tensor
            Embeddings of shape `(N, D)`.
        cls_labels : torch.Tensor
            Discrete class labels of shape `(N,)`.
        params : torch.Tensor
            Continuous parameter values of shape `(N, L)`.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """

        assert params is not None, "params must not be None for ParameterAwareTripletMarginLoss"
        assert params.ndim == 2, f"params.ndim must be 2, but got {params.ndim}"  # (N, L)

        # ----------------------------------------------------------------------------------------------------
        # Compute triplet distances and label distances

        anchor_idx, _, _, ap_dist, an_dist, _, _ = self._compute_triplet_distance(embs=embs, cls_labels=cls_labels, params=params)

        num_triplets = anchor_idx.shape[0]

        # ----------------------------------------------------------------------------------------------------
        # Compute triplet margin loss: max(0, d(a,p) - d(a,n) + margin)

        violation = ap_dist - an_dist + self._margin   # (num_triplets,)
        loss = torch.nn.functional.relu(violation)     # (num_triplets,)

        assert loss.ndim == 1, f"loss.ndim must be 1, but got {loss.ndim}"  # (N * n_pairs,)

        loss = loss.sum().div(num_triplets)  # scalar

        # ----------------------------------------------------------------------------------------------------
        # Log statistics

        if self._log_stats:
            with torch.no_grad():
                delta = an_dist - ap_dist  # (num_triplets,)

                triplet_ratio_hard = (delta <= 0.0).float().mean()
                triplet_ratio_semi_hard = ((delta > 0.0) & (delta <= self._margin)).float().mean()
                triplet_ratio_easy = (delta > self._margin).float().mean()
                triplet_nonzero_ratio = (violation > 0.0).float().mean()

                ap_dist_mean = ap_dist.mean()
                ap_dist_std = ap_dist.std(unbiased=False)
                an_dist_mean = an_dist.mean()
                an_dist_std = an_dist.std(unbiased=False)
                dist_gap_mean = (an_dist - ap_dist).mean()

                num_unique_anchors = torch.unique(anchor_idx).numel()

                # autopep8: off
                training_stats.report('metric_stats/num_triplets',       num_triplets)
                training_stats.report('metric_stats/num_unique_anchors', num_unique_anchors)
                training_stats.report('metric_stats/hard_ratio',         triplet_ratio_hard)
                training_stats.report('metric_stats/semi_hard_ratio',    triplet_ratio_semi_hard)
                training_stats.report('metric_stats/easy_ratio',         triplet_ratio_easy)
                training_stats.report('metric_stats/nonzero_ratio',      triplet_nonzero_ratio)
                training_stats.report('metric_stats/ap_dist_mean',       ap_dist_mean)
                training_stats.report('metric_stats/ap_dist_std',        ap_dist_std)
                training_stats.report('metric_stats/an_dist_mean',       an_dist_mean)
                training_stats.report('metric_stats/an_dist_std',        an_dist_std)
                training_stats.report('metric_stats/dist_gap_mean',      dist_gap_mean)
                # autopep8: on

        return loss

    @typing_extensions.override
    def forward(self, net: torch.nn.Module, imgs: torch.Tensor, cls_labels: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Forward pass of Parameter Aware Triplet Margin Loss.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network to compute embeddings.
        imgs : torch.Tensor
            Input images.
        cls_labels : torch.Tensor
            Discrete class labels for selecting samples when using certain miner types.
        params : torch.Tensor
            Continuous parameter values for computing parameter distances.

        Returns
        -------
        torch.Tensor
            Loss value.
        """

        # ----------------------------------------------------------------------------------------------------
        # Forward pass

        embs: torch.Tensor = net(imgs)  # (N, D)
        assert embs.ndim == 2, f"embs.ndim must be 2, but got {embs.ndim}"  # (N, D)

        # ----------------------------------------------------------------------------------------------------
        # Compute loss

        loss = self.forward_impl(embs=embs, cls_labels=cls_labels, params=params)

        return loss


# ----------------------------------------------------------------------------------------------------------------------------------
# Loss Catalog

loss_catalog: typing.Final[dict[str, type[MetricLossBase]]] = {
    'siamese': SiameseLoss,
    'triplet': TripletMarginLoss,
    'patriplet': ParameterAwareTripletMarginLoss,
}

# ----------------------------------------------------------------------------------------------------------------------------------
