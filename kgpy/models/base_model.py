"""
Base model class
"""
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections.abc import Iterable

from custom_embeddings import Embedding, ComplexEmbedding

if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


class Model(ABC, nn.Module):

    def __init__(
        self, 
        model_name, 
        entities, 
        relations, 
        latent_dim, 
        loss_margin, 
        
        # One of [None, 'l1', 'l2', 'l3']
        regularization, 

        # Can both be either a constant or list
        # When list 1st entry is weight for entities and 2nd for relation embeddings
        reg_weight,

        weight_init, 
        loss_fn,

        # Take norm of entities after each gradient and relations at beginning
        # TODO: Split by relation and entitiy?
        norm_constraint,
        complex_emb=False
    ):
        super(Model, self).__init__()
        
        self.name = model_name
        self.dim = latent_dim
        self.weight_init = "uniform" if weight_init is None else weight_init.lower()
        self.norm_constraint = norm_constraint

        self.entities = entities
        self.relations = relations

        self.entity_embeddings, self.relation_embeddings = self._create_embeddings(complex_emb)
        self._register_embeddings(complex_emb)

        # When not same reg weight for relation/entities you need to have only supplied 2
        if (isinstance(reg_weight, Iterable) and len(reg_weight) != 2):
            raise ValueError(f"`reg_weight` parameter must be either a constant or an iterable of length 2. You passed {reg_weight}")

        if regularization not in [None, 'l1', 'l2', 'l3']:
            raise ValueError(f"`regularization` parameter must be one of [None, 'l1', 'l2', 'l3']. You passed {regularization}")

        self.regularization = regularization
        self.reg_weight = reg_weight

        self.loss_fn_name = loss_fn.lower()

        if self.loss_fn_name == "cross-entropy":
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        elif self.loss_fn_name == "ranking":
            self.loss_fn = nn.MarginRankingLoss(margin=loss_margin, reduction='mean')
        else:
            raise ValueError(f"Invalid loss function type - {loss_fn}. Must be either 'ranking' or 'cross-entropy'")


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


    def _create_embeddings(self, complex_emb):
        """
        Create the embeddings. Control for if regular embedding or complex

        Args:
            complex_emb: bool
                True if complex

        Returns: tuple
            entity_embs, relation_embs
        """
        if complex_emb:
            entity_embeddings = ComplexEmbedding(len(self.entities), self.dim, self.weight_init).to(device)
            relation_embeddings = ComplexEmbedding(len(self.relations), self.dim, self.weight_init).to(device)
        else:
            entity_embeddings = Embedding(len(self.entities), self.dim, self.weight_init).to(device)
            relation_embeddings = Embedding(len(self.relations), self.dim, self.weight_init).to(device)

        # TODO: L1 or L2??
        if self.norm_constraint:
            relation_embeddings.normalize(1)

        return entity_embeddings, relation_embeddings


    def _register_embeddings(self, complex_emb):
        """
        Register the embeddings as part of module

        Args:
            complex_emb: bool
                True if complex

        Returns:
            None
        """
        if complex_emb:
            super(Model, self).add_module("entity_re_embs", self.entity_embeddings._emb_re._emb)
            super(Model, self).add_module("relation_re_embs", self.relation_embeddings._emb_re._emb)
            super(Model, self).add_module("entity_im_embs", self.entity_embeddings._emb_im._emb)
            super(Model, self).add_module("relation_im_embs", self.relation_embeddings._emb_im._emb)
        else:
            super(Model, self).add_module("entity_embs", self.entity_embeddings._emb)
            super(Model, self).add_module("relation_embs", self.relation_embeddings._emb)



    def forward(self, triplets, corrupted_triplets):
        """
        Forward pass for our model.
        1. Normalizes entity embeddings to unit length if specified
        2. Computes score for both types of triplets
        3. Computes loss

        Args:
            triplets: List of triplets to train on
            corrupted_triplets: Corresponding tripets with head/tail replaced

        Return:
            Return loss
        """
        if self.norm_constraint:
            self.entity_embeddings.normalize(2)

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
        reg = self._regularization()

        if self.loss_fn_name == "ranking":
            base_loss = self._ranking_loss(positive_scores, negative_scores)

        if self.loss_fn_name == "cross-entropy":
            base_loss = self._cross_entropy_loss(positive_scores, negative_scores)

        return base_loss + reg


    def _ranking_loss(self, positive_scores, negative_scores):
        """
        Compute margin ranking loss

        Args:
            positive_scores: Scores for real tiples
            negative_scores: Scores for corrupted triplets
            
        Returns:
            Loss
        """
        target = torch.ones_like(positive_scores, device=device)
        return self.loss_fn(positive_scores, negative_scores, target)


    # TODO: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # Need to implement unnormalized scores for each class (positive and negative)
    def _cross_entropy_loss(self, positive_scores, negative_scores):
        """
        Compute cross-entropy loss
        
        Args:
            positive_scores: Scores for real tiples
            negative_scores: Scores for corrupted triplets
            
        Returns:
            Loss
        """
        all_scores = torch.cat((positive_scores, negative_scores))

        target_positives = torch.ones_like(positive_scores, device=device)
        target_negatives = torch.zeros_like(negative_scores, device=device)
        all_targets = torch.cat((target_positives, target_negatives))

        return self.loss_fn(all_scores, all_targets)


    def _regularization(self):
        """
        Apply regularization if specified.

        Returns:
            Regularization term for loss
        """
        if self.regularization is None:
            return 0

        if self.regularization == "l1":
            lp = 1
        elif self.regularization == "l2":
            lp =2
        else:
            lp = 3

        entity_norm = self.entity_embeddings.norm(lp)
        relation_norm = self.relation_embeddings.norm(lp)

        if isinstance(self.reg_weight, Iterable):
            return self.reg_weight[0] * entity_norm**lp + self.reg_weight[1] * relation_norm**lp
        
        return self.reg_weight * (entity_norm**lp + relation_norm**lp) 
        