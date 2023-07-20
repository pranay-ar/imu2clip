# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    https://github.com/RElbers/info-nce-pytorch/

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(
        self,
        temperature=0.1,
        reduction="mean",
        negative_mode="unpaired",
        symmetric_loss=False,
        learn_temperature=False,
    ):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature)) if learn_temperature else temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.symmetric_loss = symmetric_loss

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(
            query,
            positive_key,
            negative_keys,
            temperature=self.temperature,
            reduction=self.reduction,
            negative_mode=self.negative_mode,
            symmetric_loss=self.symmetric_loss,
        )


def info_nce(
    query,
    positive_key,
    negative_keys=None,
    temperature=0.1,
    reduction="mean",
    negative_mode="unpaired",
    symmetric_loss=False,
):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError("<query> must have 2 dimensions.")
    if positive_key.dim() != 2:
        raise ValueError("<positive_key> must have 2 dimensions.")
    if negative_keys is not None:
        if negative_mode == "unpaired" and negative_keys.dim() != 2:
            raise ValueError(
                "<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'."
            )
        if negative_mode == "paired" and negative_keys.dim() != 3:
            raise ValueError(
                "<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'."
            )

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError(
            "<query> and <positive_key> must must have the same number of samples."
        )
    if negative_keys is not None:
        if negative_mode == "paired" and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>."
            )

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError(
            "Vectors of <query> and <positive_key> should have the same number of components."
        )
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError(
                "Vectors of <query> and <negative_keys> should have the same number of components."
            )

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    # print("Query", query)
    # print("Positive Key", positive_key)
    # print("Negative Keys", negative_keys)

    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == "unpaired":
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == "paired":
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    if symmetric_loss:
        # TODO: consider use learned temperature
        loss_i = F.nll_loss(F.log_softmax(logits / temperature, dim=0), labels)
        loss_t = F.nll_loss(F.log_softmax(logits / temperature, dim=1), labels)
        return loss_i + loss_t
    else:
        return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss for learning embeddings where the negative samples are implicitly defined
    as all samples that are not the anchor or the positive sample.
    """

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive):
        # Compute pairwise distances
        d_ap = F.pairwise_distance(anchor, positive, 2)

        # Compute implicit negative examples
        batch_size = anchor.shape[0]
        indices = torch.arange(batch_size, device=anchor.device)
        d_an = []
        for i in range(batch_size):
            neg_mask = ~torch.eq(indices, i)
            neg_samples = anchor[neg_mask]
            d_i = F.pairwise_distance(anchor[i].unsqueeze(0), neg_samples, 2)
            d_an.append(d_i.min())
        d_an = torch.stack(d_an)

        # print("Positive distance = {} | Negative distance = {}".format(d_ap, d_an))
        # Compute loss
        loss = F.relu(d_ap - d_an + self.margin).mean()

        # element_wise_loss = triplet_loss_debug(d_ap, d_an, margin=1.0)
        # print("Element-wise loss = {}".format(element_wise_loss))

        # # Plot a histogram of the element-wise loss values
        # plt.hist(element_wise_loss.cpu().numpy(), bins=50)
        # plt.xlabel('Triplet Loss')
        # plt.ylabel('Count')
        # plt.title('Histogram of Element-wise Triplet Loss')
        # plt.savefig('triplet_loss_histogram_{}.png'.format(self.margin))

        return loss
    
def triplet_loss_debug(d_ap, d_an, margin):
    loss = torch.relu(d_ap - d_an + margin)
    element_wise_loss = torch.mean(loss, dim=0)  # Calculate the loss element-wise
    return element_wise_loss
