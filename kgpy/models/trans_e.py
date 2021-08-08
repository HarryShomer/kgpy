"""
Implementation of TransE. 

See paper for more details - https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf.
"""
import torch

from .base_emb_model import SingleEmbeddingModel


class TransE(SingleEmbeddingModel):

    def __init__(
        self, 
        num_entities, 
        num_relations, 
        emb_dim=100, 
        margin=1, 
        regularization = None,
        reg_weight = 0,
        weight_init=None, 
        norm=2,
        loss_fn="ranking",
        device='cpu'
    ):
        super().__init__(
            type(self).__name__,
            num_entities, 
            num_relations, 
            emb_dim, 
            margin, 
            regularization,
            reg_weight,
            weight_init, 
            loss_fn,
            True,  # norm_constraint
            device
        )
        self.norm = norm
        

    def score_hrt(self, triplets):
        """
        Get the score for a given set of triplets.

        Negate score so that lower distances which indicates a better fit score higher than higher distances

        Parameters:
        -----------
            triplets: list
                List of triplets

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        h = self.entity_embeddings(triplets[:, 0])
        r = self.relation_embeddings(triplets[:, 1])
        t = self.entity_embeddings(triplets[:, 2])

        return - (h + r - t).norm(p=self.norm, dim=1)


    def score_head(self, triplets):
        """
        Get the score for a given set of triplets against *all possible* heads.

        Negate score so that lower distances which indicates a better fit score higher than higher distances
        
        Parameters:
        -----------
            triplets: list
                List of triplets

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        h = self.entity_embeddings(torch.arange(self.num_entities, device=self._cur_device()).long())
        r = self.relation_embeddings(triplets[:, 0])
        t = self.entity_embeddings(triplets[:, 1])

        return - (h[None, :, :] + r[:, None, :] - t[:, None, :]).norm(p=self.norm, dim=-1)


    def score_tail(self, triplets):
        """
        Get the score for a given set of triplets against *all possible* tails.

        Negate score so that lower distances which indicates a better fit score higher than higher distances

        Parameters:
        -----------
            triplets: list
                List of triplets

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        h = self.entity_embeddings(triplets[:, 1])
        r = self.relation_embeddings(triplets[:, 0])
        t = self.entity_embeddings(torch.arange(self.num_entities, device=self._cur_device()).long())

        return - (h[:, None, :] + r[:, None, :] - t[None, :, :]).norm(p=self.norm, dim=-1)
