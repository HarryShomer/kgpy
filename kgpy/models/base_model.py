"""
Base model class
"""
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


class Model(ABC, nn.Module):

    def __init__(self, model_name, entities, relations, latent_dim, loss_margin, l2, l3, weight_init):
        super(Model, self).__init__()
        
        self.name = model_name
        self.dim = latent_dim
        self.weight_init = "xavier" if weight_init is None else weight_init.lower()

        self.entities = entities
        self.relations = relations

        self.entity_embeddings, self.relation_embeddings = self._init_weights()
        self.relation_embeddings = self.relation_embeddings.to(device)

        self.l2 = l2   # L2 Regularization applied to loss (fed to optimizer)
        self.l3 = l3   # Manually added to loss
        self.loss_function = nn.MarginRankingLoss(margin=loss_margin, reduction='mean')


    @abstractmethod
    def score_function(self, triplets):
        """
        Get the score for a given set of triplets. Higher Score = More likely a true fact!

        To be implemented by the specific model.

        Args:
            triplets: List of triplets

        Returns:
            List of scores
        """
        pass



    def _init_weights(self):
        """
        Initialize the entity and relation embeddings. Initializer determined by self.weight_init

        Also divide each by relation embedding by norm.

        Returns:
            Tuple of both nn.Embedding objects
        """
        entity_embeddings = nn.Embedding(len(self.entities), self.dim)
        relation_embeddings = nn.Embedding(len(self.relations), self.dim)

        weight_init_method = self._weight_init_method()
        weight_init_method(entity_embeddings.weight)
        weight_init_method(relation_embeddings.weight)
        
        # TODO: L1 or L2??
        relation_embeddings.weight.data = self._normalize(relation_embeddings, 1)

        return entity_embeddings, relation_embeddings



    def forward(self, triplets, corrupted_triplets):
        """
        Forward pass for our model.

        1. Normalizes entity embeddings to unit length
        2. Computes score for both types of triplets
        3. Computes loss

        Args:
            triplets: List of triplets to train on
            corrupted_triplets: Corresponding tripets with head/tail replaced

        Return:
            Return loss
        """
        self.entity_embeddings.weight.data = self._normalize(self.entity_embeddings, 2)

        positive_scores = self.score_function(triplets)
        negative_scores = self.score_function(corrupted_triplets)

        return self.loss(positive_scores, negative_scores)


    def loss(self, positive_scores, negative_scores):
        """
        Compute loss

        Args:
            positive_scores: Scores for real tiples
            negative_scores: Scores for corrupted triplets

        Returns:
            Loss
        """
        if self.l3 > 0.0:
            reg = self._l3_regularization()
        else:
            reg = 0

        target = torch.ones_like(positive_scores, device=device)
        return self.loss_function(positive_scores, negative_scores, target) + reg


    def _normalize(self, embedding, p):
        """
        Normalize an embedding by some p-norm.

        Args:
            embedding: nn.Embedding object
            p: p-norm value

        Returns:
            Normalized embeddin
        """
        return embedding.weight.data / embedding.weight.data.norm(p=p, dim=1, keepdim=True)


    def _l3_regularization(self):
        """
        Commonly used for DistMult.

        See here for some l3_Weight by dataset - https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/best_config.sh
        """
        return self.l3 * (self.entity_embeddings.weight.norm(p = 3)**3 + self.relation_embeddings.weight.norm(p = 3)**3) 



    def _weight_init_method(self):
        """
        Determine the correct weight initializer method.

        Returns:
            Correct nn.init function
        """
        if self.weight_init == "normal":
            return nn.init.normal_
        elif self.weight_init == "xavier":
            return nn.init.xavier_uniform_

        raise ValueError(f"Invalid weight initializer passed {self.weight_init}. Must be either 'xavier' or 'normal'.")
