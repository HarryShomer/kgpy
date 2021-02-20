"""
Base model class
"""
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections.abc import Iterable


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
        norm_constraint
    ):
        super(Model, self).__init__()
        
        self.name = model_name
        self.dim = latent_dim
        self.weight_init = "uniform" if weight_init is None else weight_init.lower()
        self.norm_constraint = norm_constraint

        self.entities = entities
        self.relations = relations

        # When not same reg weight for relation/entities you need to have only supplied 2
        if (isinstance(reg_weight, Iterable) and len(reg_weight) != 2):
            raise ValueError(f"`reg_weight` parameter must be either a constant or an iterable of length 2. You passed {reg_weight}")

        if regularization not in [None, 'l1', 'l2', 'l3']:
            raise ValueError(f"`regularization` parameter must be one of [None, 'l1', 'l2', 'l3']. You passed {regularization}")

        self.regularization = regularization
        self.reg_weight = reg_weight

        self.loss_fn_name = loss_fn.lower()
        self.loss_fn = self._determine_loss_fn(loss_margin)


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


    @abstractmethod
    def _create_embeddings(self):
        """
        Create and initialize the embeddings.

        To be implemented by the specific type of model.

        Returns:
            Embeddings
        """
        pass


    @abstractmethod
    def _normalize_entities(self):
        """
        Normalize entity embeddings by some p-norm.

        To be implemented by the specific type of model.
        """
        pass

    @abstractmethod
    def _normalize_relations(self):
        """
        Normalize relations embeddings by some p-norm.

        To be implemented by the specific type of model.
        """
        pass


    @abstractmethod
    def _regularization(self):
        """
        Apply specific type of regularization if specified.

        To be implemented by the specific type of model.

        Returns:
            Regularization term for loss
        """
        pass


    @abstractmethod
    def _get_cur_device(self):
        """
        Get the current device being used

        Returns: str
            device name
        """
        pass


    def forward(self, triplets, corrupted_triplets):
        """
        Forward pass for our model.
        1. Normalizes entity embeddings to unit length if specified
        2. Computes score for both types of triplets
        3. Computes loss

        Args:
            triplets: list
                List of triplets to train on
            corrupted_triplets: list
                Corresponding tripets with head/tail replaced

        Returns:
            Mean loss
        """
        cur_device = self._get_cur_device()

        if self.norm_constraint:
            self._normalize_entities(2)

        positive_scores = self.score_function(triplets)
        negative_scores = self.score_function(corrupted_triplets)

        return self.loss(positive_scores, negative_scores, cur_device)


    def loss(self, positive_scores, negative_scores, device):
        """
        Compute loss

        Args:
            positive_scores: Scores for real tiples
            negative_scores: Scores for corrupted triplets
            device: optional device

        Returns:
            Loss
        """
        reg = self._regularization()

        if self.loss_fn_name == "ranking":
            base_loss = self._ranking_loss(positive_scores, negative_scores, device)

        if self.loss_fn_name == "softplus":
            base_loss = self._softplus_loss(positive_scores, negative_scores, device)

        # if self.loss_fn_name == "cross-entropy":
        #     base_loss = self._cross_entropy_loss(positive_scores, negative_scores)

        return base_loss + reg


    def _determine_loss_fn(self, loss_margin):
        """
        Determine loss function based on user input (self.loss_fn_name). 

        Throw exception when invalid.

        Args:
            loss_margin: str
                Optional margin or hinge style losses

        Returns:
            loss function methd
        """
        if self.loss_fn_name == "ranking":
            return nn.MarginRankingLoss(margin=loss_margin, reduction='mean')
        elif self.loss_fn_name == "bce":
            return nn.BCEWithLogitsLoss(reduction='mean')
        
        # TODO
        elif self.loss_fn_name == "softplus":
            return
      
        raise ValueError(f"Invalid loss function type - {loss_fn}. Must be either 'ranking' or 'cross-entropy'")
        

    def _get_weight_init_method(self):
        """
        Determine the correct weight initializer method and init weights

        Args:
            weight_init_method: str
                Type of weight init method. Currently only works with "uniform" and "normal"

        Returns:
            Correct nn.init function
        """
        if self.weight_init == "normal":
            return nn.init.xavier_normal_
        elif self.weight_init == "uniform":
            return nn.init.xavier_uniform_
        else:
            raise ValueError(f"Invalid weight initializer passed {self.weight_init}. Must be either 'uniform' or 'normal'.")


    def _ranking_loss(self, positive_scores, negative_scores, device):
        """
        Compute margin ranking loss

        Args:
            positive_scores: Scores for real tiples
            negative_scores: Scores for corrupted triplets
            device: optional device
            
        Returns:
            Loss
        """
        target = torch.ones_like(positive_scores, device=device)
        return self.loss_fn(positive_scores, negative_scores, target)


    def _softplus_loss(self, positive_scores, negative_scores, device):
        """
        Minimize the softplus loss.

        L(score_i, label_i) = log(1 + exp(-label_i * score_i))

        Example: Used in Complex

        Args:
            positive_scores: Scores for real tiples
            negative_scores: Scores for corrupted triplets
            
        Returns:
            Loss
        """
        softplus = nn.Softplus(beta=1)

        positive_scores *= -1
        all_scores = torch.cat((positive_scores, negative_scores))

        return softplus(all_scores).mean()



    def _bce_loss(self, positive_scores, negative_scores, device):
        """
        Compute Binary coss entropy loss

        Args:
            positive_scores: Scores for real tiples
            negative_scores: Scores for corrupted triplets
            device: optional device
        
        Returns:
            Loss
        """
        all_scores = torch.cat((positive_scores, negative_scores))

        target_positives = torch.ones_like(positive_scores, device=device)
        target_negatives = torch.zeros_like(negative_scores, device=device)
        all_targets = torch.cat((target_positives, target_negatives))

        return self.loss_fn(all_scores, all_targets)



    # TODO - Worth doing?
    def _cross_entropy_loss(self, positive_scores, negative_scores, device):
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



    def _normalize(self, emb, p):
        """
        Normalize an embedding by some p-norm.

        Args:
            emb: nn.Embedding
            p: p-norm value

        Returns:
            Embedding
        """
        emb.weight.data = emb.weight.data / self._norm(emb, p, dim=1, keepdim=True)
        return emb


    def _norm(self, emb, p, **kwargs):
        """
        Return norm of the embeddings

        Args:
            emb: nn.Embedding
            p: p-norm value

        Returns:
            Norm value
        """
        return emb.weight.data.norm(p=p, **kwargs)


#####################################################
#####################################################
#####################################################


class SingleEmbeddingModel(Model):
    """
    Each entity / relation gets one embedding
    """
    def __init__(
        self, 
        model_name, 
        entities, 
        relations, 
        latent_dim, 
        loss_margin, 
        regularization, 
        reg_weight,
        weight_init, 
        loss_fn,
        norm_constraint
    ):
        super().__init__(
            model_name, 
            entities, 
            relations, 
            latent_dim, 
            loss_margin, 
            regularization, 
            reg_weight,
            weight_init, 
            loss_fn,
            norm_constraint
        )
        self.entity_embeddings, self.relation_embeddings = self._create_embeddings()

        # TODO: L1 or L2??
        if self.norm_constraint:
           self._normalize_relations(1)


    def _create_embeddings(self):
        """
        Create the embeddings. Control for if regular embedding or complex

        Args:
            complex_emb: bool
                True if complex

        Returns: tuple
            entity_embs, relation_embs
        """
        weight_init_method = self._get_weight_init_method()

        entity_emb = nn.Embedding(len(self.entities), self.dim)
        relation_emb = nn.Embedding(len(self.entities), self.dim)

        weight_init_method(entity_emb.weight)
        weight_init_method(relation_emb.weight)

        return entity_emb, relation_emb


    def _normalize_entities(self, p):
        """
        Normalize entity embeddings by some p-norm. Does so in-place

        Args:
            p: p-norm value

        Returns:
            None
        """
        self.entity_embeddings = self._normalize(self.entity_embeddings, p)

    
    def _normalize_relations(self, p):
        """
        Normalize relations embeddings by some p-norm.  Does so in-place

        Args:
            p: p-norm value

        Returns:
            Norne
        """
        self.relation_embeddings = self._normalize(self.relation_embeddings, p)


    def _regularization(self):
        """
        Apply regularization if specified.
        Returns:
            Regularization term for loss
        """
        if self.regularization is None:
            return 0

        lp = int(self.regularization[1])
        entity_norm = self._norm(self.entity_embeddings, lp)
        relation_norm = self._norm(self.relation_embeddings, lp)

        if isinstance(self.reg_weight, Iterable):
            return self.reg_weight[0] * entity_norm**lp + self.reg_weight[1] * relation_norm**lp
        
        return self.reg_weight * (entity_norm**lp + relation_norm**lp) 


    def _get_cur_device(self):
        """
        Get the current device being used

        Returns: str
            device name
        """
        return self.entity_embeddings.weight.device


#####################################################
#####################################################
#####################################################


class ComplexEmbeddingModel(Model):
    """
    """
    def __init__(
        self, 
        model_name, 
        entities, 
        relations, 
        latent_dim, 
        loss_margin, 
        regularization, 
        reg_weight,
        weight_init, 
        loss_fn,
        norm_constraint,
    ):
        super().__init__(
            model_name, 
            entities, 
            relations, 
            latent_dim, 
            loss_margin, 
            regularization, 
            reg_weight,
            weight_init, 
            loss_fn,
            norm_constraint
        )
        
        self.entity_emb_re, self.entity_emb_im, self.relation_emb_re, self.relation_emb_im = self._create_embeddings()
        
        # TODO: L1 or L2??
        if self.norm_constraint:
           self._normalize_relations(1)


    def _create_embeddings(self):
        """
        Create the complex embeddings.

        Args:
            complex_emb: bool
                True if complex

        Returns: tuple
            entity_embs, relation_embs
        """
        entity_embs, relation_embs = [], []
        weight_init_method = self._get_weight_init_method()

        entity_emb_re = nn.Embedding(len(self.entities), self.dim)
        relation_emb_re = nn.Embedding(len(self.entities), self.dim)
        entity_emb_im = nn.Embedding(len(self.entities), self.dim)
        relation_emb_im = nn.Embedding(len(self.entities), self.dim)

        weight_init_method(entity_emb_re.weight)
        weight_init_method(relation_emb_re.weight)
        weight_init_method(entity_emb_im.weight)
        weight_init_method(relation_emb_im.weight)

        return entity_emb_re, entity_emb_im, relation_emb_re, relation_emb_im


    def _normalize_entities(self, p):
        """
        Normalize entity embeddings by some p-norm. Does so in-place

        Args:
            p: p-norm value

        Returns:
            None
        """
        self.entity_emb_re = self._normalize(self.entity_emb_re, p)
        self.entity_emb_im = self._normalize(self.entity_emb_im, p)


    def _normalize_relations(self, p):
        """
        Normalize relations embeddings by some p-norm.  Does so in-place

        Args:
            p: p-norm value

        Returns:
            None
        """
        self.relation_emb_re = self._normalize(self.relation_emb_re, p)
        self.relation_emb_im = self._normalize(self.relation_emb_im, p)


    def _regularization(self):
        """
        Apply regularization if specified.

        Returns:
            Regularization term for loss
        """
        if self.regularization is None:
            return 0

        lp = int(self.regularization[1])
        
        entity_re = self._norm(self.entity_emb_re, lp)
        entity_im = self._norm(self.entity_emb_im, lp)
        relation_re = self._norm(self.relation_emb_re, lp)
        relation_im = self._norm(self.relation_emb_im, lp)


        if isinstance(self.reg_weight, Iterable):
            return self.reg_weight[0] * (entity_re**lp + entity_im**lp) + self.reg_weight[1] * (relation_re**lp + relation_im**lp)
        
        return self.reg_weight * (entity_re**lp + entity_im**lp + relation_re**lp + relation_im**lp) 


    def _get_cur_device(self):
        """
        Get the current device being used

        Returns: str
            device name
        """
        return self.entity_emb_re.weight.device