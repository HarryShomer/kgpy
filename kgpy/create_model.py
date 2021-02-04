"""
Implementation of TransE. 

See paper for more details - https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf.
"""
import copy
import numpy as np
from random import randint, choice
import torch
import torch.nn as nn

if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


class TransE(nn.Module):

    def __init__(self, entities, relations, latent_dim=100, loss_margin=1, norm=1):
        super(TransE, self).__init__()
        self.name = "TransE"
        
        self.dim = latent_dim
        self.norm = norm

        self.entities = entities
        self.relations = relations

        self.entity_embeddings, self.relation_embeddings = self._init_weights()
        self.relation_embeddings = self.relation_embeddings.to(device)

        self.loss_function = nn.MarginRankingLoss(margin=loss_margin, reduction='mean')


    def _init_weights(self):
        """
        Initialize the entity and relation embeddings. Range if uniform btwn +/- 6/sqrt(dim).

        Also divide each by relation embedding norm as specified in paper.

        Returns:
            Tuple of both nn.Embedding objects
        """
        entity_embeddings = nn.Embedding(len(self.entities), self.dim)
        relation_embeddings = nn.Embedding(len(self.relations), self.dim)
        
        weight_range = 6 / np.sqrt(self.dim)
        nn.init.uniform_(entity_embeddings.weight, -weight_range, weight_range)
        nn.init.uniform_(relation_embeddings.weight, -weight_range, weight_range)

        # TODO: L1 or L2??
        relation_embeddings.weight.data = relation_embeddings.weight.data / relation_embeddings.weight.data.norm(p=1, dim=1, keepdim=True)

        return entity_embeddings, relation_embeddings



    def forward(self, triplets, corrupted_triplets):
        """
        Forward pass for our model.

        1. Normalizes entity embeddings
        2. Computes score for both types of triplets
        3. Computes loss

        Args:
            triplets: List of triplets to train on
            corrupted_triplets: Corresponding tripets with head/tail replaced

        Return:
            Return loss
        """
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data / self.entity_embeddings.weight.data.norm(p=2, dim=1, keepdim=True)

        positive_scores = self._predict(triplets)
        negative_scores = self._predict(corrupted_triplets)

        return self.loss(positive_scores, negative_scores)



    def _predict(self, triplets):
        """
        Pass list of tripets through score function for loss.

        Because the margin loss expects the true pair to be greater than incorrect pair we negate the score function.
        Since the more true a triplet is the smaller the distance

        Args:
            triplets: List of triplets

        Returns:
            List of scores
        """
        return - self.score_function(triplets)


    def loss(self, positive_scores, negative_scores):
        """
        Compute loss

        Args:
            positive_scores: Scores for real tiples
            negative_scores: Scores for corrupted triplets

        Returns:
            Loss
        """
        target = torch.ones_like(positive_scores, device=device)
        return self.loss_function(positive_scores, negative_scores, target)


    def score_function(self, triplets):
        """
        Get the score for a given set of triplets.

        Args:
            triplets: List of triplets

        Returns:
            List of scores
        """
        h = triplets[:, 0]
        r = triplets[:, 1]
        t = triplets[:, 2]

        return (self.entity_embeddings(h) + self.relation_embeddings(r) - self.entity_embeddings(t)).norm(p=self.norm, dim=1)


    def corrupt_triplets(self, triplets):
        """
        Corrupt list of triplet by randomly replacing either the head or the tail with another entitiy

        Args:
            triplets: list of triplet to corrupt 

        Returns:
            Corrupted Triplets
        """
        corrupted_triplets = copy.deepcopy(triplets)

        for i, t in enumerate(triplets):
            head_tail = choice([0, 2])
            corrupted_triplets[i][head_tail] = self.randint_exclude(0, len(self.entities), t[head_tail])

        return corrupted_triplets


    def randint_exclude(self, begin, end, exclude):
        """
        Randint but exclude a number

        Args:
            begin: begin of range
            end: end of range (exclusive)
            exclude: number to exclude

        Returns:
            randint not in exclude
        """
        while True:
            x = randint(begin, end-1)

            if x != exclude:
                return x
