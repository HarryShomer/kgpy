"""
TODO: Provide interface here to better manage loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class Loss(ABC, nn.Module):
    """
    Base loss class specification
    """
    def __init__(self):
        super(Loss, self).__init__()


    @abstractmethod
    def forward(self):
        """
        Compute loss on sample.

        To be implemented by specific loss function.
        """
        pass


class MarginRankingLoss(Loss):
    """
    Wrapper for Margin Ranking Loss
    """

    def __init__(self, margin):
        """
        Constructor

        Parameters:
        ----------
            margin: int
                loss margin
        """
        super().__init__()
        self.margin = margin


    def forward(self, positive_scores, negative_scores, device="cpu"):
        """
        Compute loss on sample.

        Parameters:
        -----------
            positive_scores: Tensor
                Scores for true triplets
            negative_scores: Tensor
                Scores for corrupted triplets
            device: str
                device being used. defaults to "cpu"

        Returns:
        --------
        float
            loss
        """
        target = torch.ones_like(positive_scores, device=device)
        return F.margin_ranking_loss(positive_scores, negative_scores, target, margin=self.margin, reduction='mean')



class BCELoss(Loss):
    """
    Wrapper for Binary cross entropy loss (includes logits)
    """
    def __init__(self):
        super().__init__()


    def forward(self, positive_scores, negative_scores, device="cpu"):
        """
        Compute loss on sample.

        Parameters:
        -----------
            positive_scores: Tensor
                Scores for true triplets
            negative_scores: Tensor
                Scores for corrupted triplets
            device: str
                device being used. defaults to "cpu"

        Returns:
        --------
        float
            loss
        """
        all_scores = torch.cat((positive_scores, negative_scores))

        target_positives = torch.ones_like(positive_scores, device=device)
        target_negatives = torch.zeros_like(negative_scores, device=device)
        all_targets = torch.cat((target_positives, target_negatives))

        return F.binary_cross_entropy_with_logits(all_scores, all_targets, reduction='mean')



class SoftPlusLoss(Loss):
    """
    Wrapper for Softplus loss (used for ComplEx)
    """
    def __init__(self):
        super().__init__()


    def forward(self, positive_scores, negative_scores, device="cpu"):
        """
        Compute loss on sample.

        Parameters:
        -----------
            positive_scores: Tensor
                Scores for true triplets
            negative_scores: Tensor
                Scores for corrupted triplets
            device: str
                device being used. defaults to "cpu"

        Returns:
        --------
        float
            loss
        """
        positive_scores *= -1
        all_scores = torch.cat((positive_scores, negative_scores))

        return F.softplus(all_scores, beta=1).mean() 


def NegativeSamplingLoss(Loss):
    """
    See RotatE paper
    """
    def __init__(self, margin):
        super().__init__()
        self.margin = margin


    def forward(self, positive_scores, negative_scores, negative_weights, device="cpu"):
        """
        Compute loss on sample.

        Parameters:
        -----------
            positive_scores: Tensor
                Scores for true triplets
            negative_scores: Tensor
                Scores for corrupted triplets
            negative_weights: Tensor
                Weights for negative_scores
            device: str
                device being used. defaults to "cpu"

        Returns:
        --------
        float
            loss
        """
        # TODO: Make sure work on batch level
        pos_score = - F.logsigmoid(self.margin - positive_scores)

        # TODO: Sum - batch?
        neg_score = negative_weights * F.logsigmoid(negative_scores - self.margin)

        #return (pos_score - neg_score).mean()
