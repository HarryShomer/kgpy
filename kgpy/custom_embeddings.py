import torch
import torch.nn as nn


class Embedding:
    """
    Wrapper for nn.Embedding. Mostly just passes stuff on.

    Implemented to make dealing with Complex Embeddings easier

    TODO: Subclass nn.Embedding?
    """
    def __init__(self, num_embeddings, embedding_dim, weight_init_method, **kwargs):
        self._emb = nn.Embedding(num_embeddings, embedding_dim, **kwargs)
        self._init_weights(weight_init_method)


    @property
    def num_embeddings(self):
        return self._emb.num_embeddings


    @property
    def embedding_dim(self):
        return self._emb.embedding_dim


    def __call__(self, *args):
        """
        Pass off to embedding obj
        """
        return self._emb(*args)


    def to(self, device):
        """
        Register device
        """
        self._emb.to(device)
        return self


    def normalize(self, p):
        """
        Normalize an embedding by some p-norm.

        Args:
            p: p-norm value

        Returns:
            None
        """
        self._emb.weight.data = self._emb.weight.data / self.norm(p, dim=1, keepdim=True)


    def norm(self, p, **kwargs):
        """
        Return norm of the embeddings

        Args:
            p: p-norm value

        Returns:
            Norm value
        """
        return self._emb.weight.data.norm(p=p, **kwargs)


    def _init_weights(self, weight_init_method):
        """
        Determine the correct weight initializer method and init weights

        Args:
            weight_init_method: str
                Type of weight init method. Currently only works with "uniform" and "normal"

        Returns:
            Correct nn.init function
        """
        if weight_init_method == "normal":
            nn.init.xavier_normal_(self._emb.weight)
        elif weight_init_method == "uniform":
            nn.init.xavier_uniform_(self._emb.weight)
        else:
            raise ValueError(f"Invalid weight initializer passed {weight_init_method}. Must be either 'uniform' or 'normal'.")
        


class ComplexEmbedding:
    """
    Holds 2 Embedding objects.

    Implements appropriate functions so base_model.Model can interface with it
    """
    def __init__(self, num_embeddings, embedding_dim, weight_init_method, **kwargs):
        self._emb_re = Embedding(num_embeddings, embedding_dim, weight_init_method, **kwargs)
        self._emb_im = Embedding(num_embeddings, embedding_dim, weight_init_method, **kwargs)
        

    @property
    def num_embeddings(self):
        return self._emb_re.num_embeddings


    @property
    def embedding_dim(self):
        return self._emb_re.embedding_dim + self._emb_im.embedding_dim


    def __call__(self, *args):
        """
        Pass off to embedding objs

        Returns: tuple
            real embedding, imaginary embedding
        """
        return self._emb_re(*args), self._emb_im(*args)


    def to(self, device):
        """
        Register device
        """
        self._emb_re.to(device)
        self._emb_im.to(device)
        
        return self


    def normalize(self, p):
        """
        Normalize an embedding by some p-norm.

        Args:
            p: p-norm value

        Returns:
            None
        """
        self._emb_re.normalize(p)
        self._emb_im.normalize(p)


    def norm(self, p, **kwargs):
        """
        Return norm of the embeddings

        Args:
            p: p-norm value

        Returns:
            Norm value for each component
        """
        return self._emb_re.norm(p, **kwargs), self._emb_im.norm(p, **kwargs)


        