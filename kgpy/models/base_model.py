"""
Base model class
"""
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections.abc import Iterable

from kgpy import loss


class Model(ABC, nn.Module):
    """
    Base Model Class

    Attributes:
    -----------
    name: str
        Name of model
    entities: list
        list of entities (`entities` attribute of `loa_data.AllDataSet` object) 
    relations: list
        list of relations (`relations` attribute of `loa_data.AllDataSet` object)
    dim: int
        hidden dimension
    regularization: str 
        Type of regularization. One of [None, 'l1', 'l2', 'l3']
    reg_weight: list/float
        Regularization weights. When list 1st entry is weight for entities and 2nd for relation embeddings.
    weight_init: str
        weight_init method to use
    loss_fn: loss.Loss
        Loss function object
    norm_constraint: bool
        Whether Take norm of entities after each gradient and relations at beginning   
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
        norm_constraint   # TODO: Split by relation and entitiy? Also allow specfication of norm?
    ):
        """
        Model constructor

        Parameters:
        -----------
            model_name: str
                Name of model
            entities: list
                list of entities (`entities` attribute of `loa_data.AllDataSet` object) 
            relations: list
                list of relations (`relations` attribute of `loa_data.AllDataSet` object)
            latent_dim: int
                hidden dimension
            loss_margin: int
                margin to use if using a margin-based loss
            regularization: str 
                Type of regularization. One of [None, 'l1', 'l2', 'l3']
            reg_weight: list/float
                Regularization weights. When list 1st entry is weight for entities and 2nd for relation embeddings.
            weight_init: str
                weight_init method to use
            loss_fn: str
                name of loss function to use
            norm_constraint: bool
                Whether Take norm of entities after each gradient and relations at beginning       

        Returns:
        --------
        None
        """

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

        self.loss_fn = self._determine_loss_fn(loss_fn.lower(), loss_margin)


    @abstractmethod
    def score_function(self, triplets):
        """
        Get the score for a given set of triplets. Higher Score = More likely a true fact!

        To be implemented by the specific model.

        Parameters:
        -----------
            triplets: List of triplets

        Returns:
        --------
            List of scores
        """
        pass


    @abstractmethod
    def _create_embeddings(self):
        """
        Create and initialize the embeddings.

        To be implemented by the specific type of model.

        Returns:
        --------
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
        --------
            Regularization term for loss
        """
        pass


    @abstractmethod
    def _get_cur_device(self):
        """
        Get the current device being used

        To be implemented by the specific type of model.
        
        Returns:
        --------
        str
            device name
        """
        pass


    def forward(self, triplets, corrupted_triplets):
        """
        Forward pass for our model.
        1. Normalizes entity embeddings to unit length if specified
        2. Computes score for both types of triplets
        3. Computes loss

        Parameters:
        -----------
            triplets: list
                List of triplets to train on
            corrupted_triplets: list
                Corresponding tripets with head/tail replaced

        Returns:
        --------
            Mean loss
        """
        cur_device = self._get_cur_device()

        if self.norm_constraint:
            self._normalize_entities(2)

        pos_scores = self.score_function(triplets)
        neg_scores = self.score_function(corrupted_triplets)

        return self.loss_fn(pos_scores, neg_scores, cur_device) + self._regularization()



    # TODO: Move outside of class?
    def _determine_loss_fn(self, loss_fn_name, loss_margin):
        """
        Determine loss function based on user input (self.loss_fn_name). 

        Throw exception when invalid.

        Parameters:
        -----------
            loss_fn_name: str
                Name of loss function fed to constructor
            loss_margin: int
                Optional margin or hinge style losses

        Returns:
        --------
        loss.Loss
            Specific loss function
        """
        if loss_fn_name == "ranking":
            return loss.MarginRankingLoss(margin=loss_margin)
        elif loss_fn_name == "bce":
            return loss.BCELoss()
        elif loss_fn_name == "softplus":
            return loss.SoftPlusLoss()
      
        raise ValueError(f"Invalid loss function type - {loss_fn_name}")
        

    def _get_weight_init_method(self):
        """
        Determine the correct weight initializer method and init weights

        Parameters:
        -----------
            weight_init_method: str
                Type of weight init method. Currently only works with "uniform" and "normal"

        Returns:
        --------
            Correct nn.init function
        """
        if self.weight_init == "normal":
            return nn.init.xavier_normal_
        elif self.weight_init == "uniform":
            return nn.init.xavier_uniform_
        else:
            raise ValueError(f"Invalid weight initializer passed {self.weight_init}. Must be either 'uniform' or 'normal'.")


    def _normalize(self, emb, p):
        """
        Normalize an embedding by some p-norm.

        Parameters:
        -----------
            emb: nn.Embedding
            p: p-norm value

        Returns:
        --------
            Embedding
        """
        emb.weight.data = emb.weight.data / self._norm(emb, p, dim=1, keepdim=True)
        return emb


    def _norm(self, emb, p, **kwargs):
        """
        Return norm of the embeddings

        Parameters:
        -----------
            emb: nn.Embedding
            p: p-norm value

        Returns:
        --------
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

        if self.norm_constraint:
           self._normalize_relations(2)


    def _create_embeddings(self):
        """
        Create the embeddings. Control for if regular embedding or complex

        Parameters:
        -----------
            complex_emb: bool
                True if complex

        Returns:
        --------
        tuple
            entity_embs, relation_embs
        """
        weight_init_method = self._get_weight_init_method()

        entity_emb = nn.Embedding(len(self.entities), self.dim)
        relation_emb = nn.Embedding(len(self.relations), self.dim)

        weight_init_method(entity_emb.weight)
        weight_init_method(relation_emb.weight)

        return entity_emb, relation_emb


    def _normalize_entities(self, p):
        """
        Normalize entity embeddings by some p-norm. Does so in-place

        Parameters:
        -----------
            p: int
                p-norm value

        Returns:
        --------
            None
        """
        self.entity_embeddings = self._normalize(self.entity_embeddings, p)


    
    def _normalize_relations(self, p):
        """
        Normalize relations embeddings by some p-norm.  Does so in-place

        Parameters:
        -----------
            p: int
                p-norm value

        Returns:
        --------
            Norne
        """
        self.relation_embeddings = self._normalize(self.relation_embeddings, p)


    def _regularization(self):
        """
        Apply regularization if specified.

        Returns:
        --------
        float
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

        Returns:
        --------
        str
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
        
        if self.norm_constraint:
           self._normalize_relations(2)


    def _create_embeddings(self):
        """
        Create the complex embeddings.

        Parameters:
        -----------
            complex_emb: bool
                True if complex

        Returns:
        --------
        tuple
            entity_emb_re, entity_emb_im, relation_emb_re, relation_emb_im
        """
        weight_init_method = self._get_weight_init_method()

        entity_emb_re = nn.Embedding(len(self.entities), self.dim)
        relation_emb_re = nn.Embedding(len(self.relations), self.dim)
        entity_emb_im = nn.Embedding(len(self.entities), self.dim)
        relation_emb_im = nn.Embedding(len(self.relations), self.dim)

        weight_init_method(entity_emb_re.weight)
        weight_init_method(relation_emb_re.weight)
        weight_init_method(entity_emb_im.weight)
        weight_init_method(relation_emb_im.weight)

        return entity_emb_re, entity_emb_im, relation_emb_re, relation_emb_im


    def _normalize_entities(self, p):
        """
        Normalize entity embeddings by some p-norm. Does so in-place.

        Parameters:
        -----------
            p: int
                p-norm value

        Returns:
        --------
            None
        """
        e_re_sum = self.entity_emb_re.weight.pow(p).sum(dim=-1)
        e_im_sum = self.entity_emb_im.weight.pow(p).sum(dim=-1)
        
        e_norm = torch.sqrt(e_re_sum + e_im_sum)

        self.entity_emb_re.weight.data = self.entity_emb_re.weight.data / e_norm.reshape(-1, 1)
        self.entity_emb_im.weight.data = self.entity_emb_im.weight.data / e_norm.reshape(-1, 1)


    def _normalize_relations(self, p):
        """
        Normalize relations embeddings by some p-norm.  Does so in-place

        Parameters:
        -----------
            p: int
                p-norm value

        Returns:
        --------
            None
        """
        r_re_sum = self.relation_emb_re.weight.pow(p).sum(dim=-1)
        r_im_sum = self.relation_emb_im.weight.pow(p).sum(dim=-1)
        
        r_norm = torch.sqrt(r_re_sum + r_im_sum)

        self.relation_emb_re.weight.data = self.relation_emb_re.weight.data / r_norm.reshape(-1, 1)
        self.relation_emb_im.weight.data = self.relation_emb_im.weight.data / r_norm.reshape(-1, 1)



    def _regularization(self):
        """
        Apply regularization if specified.

        Returns:
        --------
        float
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

        Returns:
        --------
        str
            device name
        """
        return self.entity_emb_re.weight.device