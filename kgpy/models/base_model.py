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

    def __init__(self, model_name, entities, relations, latent_dim, loss_margin, l2, l3, init_weight_range):
        super(Model, self).__init__()
        
        self.name = model_name
        self.dim = latent_dim

        self.entities = entities
        self.relations = relations

        self.entity_embeddings, self.relation_embeddings = self._init_weights(init_weight_range)
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


    def _create_weight_init_range(self, init_weight_range):
        """
        If a weight range is give use that. Otherwise default to xavier.

        Always assume uniform. Also if invalid range use xavier.

        Args:
            init_weight_range: Range given by user. Should be a iterable with 2 elements

        Return:
            List with to elements of range
        """
        xavier = [- 6 / np.sqrt(self.dim), 6 / np.sqrt(self.dim)]

        if not isinstance(init_weight_range, list) or len(init_weight_range) == 0:
            return xavier 
        elif len(init_weight_range) != 2:
            print(f"Invalid weight range given for weights {init_weight_range}. Defaulting to xavier init.")
            return xavier
        else:
            return init_weight_range



    def _init_weights(self, init_weight_range):
        """
        Initialize the entity and relation embeddings. Range if uniform btwn +/- 6/sqrt(dim).

        Also divide each by relation embedding norm as specified in paper.

        Args:
            init_weight_range: Range supplied by user. See function self._create_weight_init_range for more details.

        Returns:
            Tuple of both nn.Embedding objects
        """
        entity_embeddings = nn.Embedding(len(self.entities), self.dim)
        relation_embeddings = nn.Embedding(len(self.relations), self.dim)
        
        weight_range = self._create_weight_init_range(init_weight_range)
        nn.init.uniform_(entity_embeddings.weight, weight_range[0], weight_range[1])
        nn.init.uniform_(relation_embeddings.weight, weight_range[0], weight_range[1])

        # TODO: L1 or L2??
        relation_embeddings.weight.data = self.normalize(relation_embeddings, 1)

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
        self.entity_embeddings.weight.data = self.normalize(self.entity_embeddings, 2)

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
            reg = self.l3_regularization()
        else:
            reg = 0

        target = torch.ones_like(positive_scores, device=device)
        return self.loss_function(positive_scores, negative_scores, target) + reg


    def normalize(self, embedding, p):
        """
        Normalize an embedding by some p-norm.

        Args:
            embedding: nn.Embedding object
            p: p-norm value

        Returns:
            Normalized embeddin
        """
        return embedding.weight.data / embedding.weight.data.norm(p=p, dim=1, keepdim=True)


    def l3_regularization(self):
        """
        Commonly used for DistMult.

        See here for some l3_Weight by dataset - https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/best_config.sh
        """
        return self.l3 * (self.entity_embeddings.weight.norm(p = 3)**3 + self.relation_embeddings.weight.norm(p = 3)**3) 