# modified from https://github.com/jozhang97/DETA/blob/master/models/assigner.py
from typing import List

import torch
import torch.nn as nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou, box_xyxy_to_cxcywh


def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)


def subsample_labels(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx


def sample_topk_per_gt(pr_inds, gt_inds, cost_matrix, k):
    """
    pr_inds (tensor): tensor of shape (M,)
    gt_inds (tensor): tensor of shape (M,)
    cost_matrix (tensor): tensor of shape (num_targets, num_queries)
    """
    if len(gt_inds) == 0:
        return pr_inds, gt_inds
    # find topk matches for each gt
    gt_inds2, counts = gt_inds.unique(return_counts=True)
    scores, pr_inds2 = cost_matrix[gt_inds2].topk(k, dim=1)
    gt_inds2 = gt_inds2[:,None].repeat(1, k)

    # filter to as many matches that gt has
    pr_inds3 = torch.cat([pr[:c] for c, pr in zip(counts, pr_inds2)])
    gt_inds3 = torch.cat([gt[:c] for c, gt in zip(counts, gt_inds2)])
    scores = torch.cat([s[:c] for c, s in zip(counts, scores)])
    
    # assign query to gt with highest match score
    score_sorted_inds = scores.argsort(descending=False)
    pr_inds3 = pr_inds3[score_sorted_inds]
    gt_inds3 = gt_inds3[score_sorted_inds]

    return pr_inds3, gt_inds3


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.

    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.

    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(
        self, thresholds: List[float], labels: List[int], allow_low_quality_matches: bool = False
    ):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.

            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        # Add -inf and +inf to first and last position in thresholds
        thresholds = thresholds[:]
        # assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        # Currently torchscript does not support all + generator
        assert all([low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])]), thresholds
        assert all([l in [-1, 0, 1] for l in labels])
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )
            # When no gt boxes exist, we define IOU = 0 and therefore set labels
            # to `self.labels[0]`, which usually defaults to background class 0
            # To choose to ignore instead, can make labels=[-1,0,-1,1] + set appropriate thresholds
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix, k=1):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of
        :paper:`Faster R-CNN`.
        """
        highest_quality_foreach_gt_inds = match_quality_matrix.topk(k=k, dim=1)[1]
        match_labels[highest_quality_foreach_gt_inds.flatten()] = 1


# modified from https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/modeling/roi_heads/roi_heads.py#L123
class Stage2Assigner(nn.Module):
    def __init__(self, threshold=0.4, k=6, coef_box=0.7, coef_cls=0.3):
        super().__init__()
        self.positive_fraction = 0.25
        self.bg_label = 400  # number > 91 to filter out later
        # self.batch_size_per_image = num_queries
        self.k = k
        self.coef_box = coef_box
        self.coef_cls = coef_cls

        self.proposal_matcher = Matcher(thresholds=[threshold], labels=[0, 1], allow_low_quality_matches=True)

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor, batch_size_per_image: int
    ):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.bg_label
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.bg_label
        
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, batch_size_per_image, self.positive_fraction, self.bg_label
        )
        
        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]
    
    def _process_proposals(self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor):
        has_gt = gt_classes.numel() > 0
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_classes[matched_labels == 0] = self.bg_label
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.bg_label
        
        return gt_classes

    def _select_topk_per_gt(self, pr_inds, gt_inds, cost_matrix, max_k=6):
        if len(gt_inds) == 0:
            return pr_inds, gt_inds

        scores = cost_matrix[gt_inds, pr_inds]
        final_pr_inds = []
        final_gt_inds = []
        for gt_idx in gt_inds.unique():
            indices = torch.argwhere(gt_inds == gt_idx).flatten()
            selected_pr = pr_inds[indices]
            selected_scores = scores[indices]
            if len(indices) < max_k:
                final_pr_inds.append(selected_pr)
                final_gt_inds.append(gt_idx.repeat(len(indices)))
                continue
            topk_indices = selected_scores.topk(k=max_k)[1]
            selected_pr = selected_pr[topk_indices]
            final_pr_inds.append(selected_pr)
            final_gt_inds.append(gt_idx.repeat(max_k))

        return torch.cat(final_pr_inds), torch.cat(final_gt_inds)

    def postprocess_indices(self, pr_inds, gt_inds, iou, k):
        return sample_topk_per_gt(pr_inds, gt_inds, iou, k)

    @torch.no_grad()
    def get_cost_matrix(self, pred_logits, pred_boxes, gt_classes, gt_boxes):
        num_queries = len(pred_logits)
        out_prob = pred_logits.sigmoid()
        out_bbox = pred_boxes

        cost_box = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(gt_boxes))[0]
        
        cost_class = out_prob[:, gt_classes]

        C = self.coef_box * cost_box + self.coef_cls * cost_class
        C = C.view(num_queries, -1)

        return C.T

    def forward(self, outputs, targets, return_cost_matrix=False):
        # COCO categories are from 1 to 90. They set num_classes=91 and apply sigmoid.
        bs, num_queries = outputs['pred_logits'].shape[:2]

        indices = []
        cost_matrices = []

        with torch.no_grad():
            for b in range(bs):
                pred_logits = outputs['pred_logits'][b].detach()
                pred_boxes = outputs['pred_boxes'][b]
                gt_boxes = targets[b]['boxes']
                gt_classes = targets[b]['labels']

                cost_matrix = self.get_cost_matrix(pred_logits, pred_boxes, gt_classes, gt_boxes)

                matched_idxs, matched_labels = self.proposal_matcher(cost_matrix)
                sampled_idxs, sampled_gt_classes = self._sample_proposals(
                    matched_idxs, matched_labels, targets[b]['labels'], batch_size_per_image=num_queries
                )
                pos_pr_inds = sampled_idxs[sampled_gt_classes != self.bg_label]
                pos_gt_inds = matched_idxs[pos_pr_inds]

                pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, cost_matrix, self.k)
                indices.append((pos_pr_inds, pos_gt_inds))
                cost_matrices.append(cost_matrix)

        if return_cost_matrix:
            return indices, cost_matrices
        return indices
