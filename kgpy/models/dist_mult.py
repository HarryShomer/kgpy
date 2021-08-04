"""
Implementation of DistMult. 

See paper for more details - https://arxiv.org/pdf/1412.6575.pdf.
"""
import torch

from .base_emb_model import SingleEmbeddingModel


class DistMult(SingleEmbeddingModel):
    def __init__(self, 
        num_entities, 
        num_relations, 
        emb_dim=100, 
        margin=1, 
        regularization = 'l3',
        reg_weight = 1e-6,
        weight_init=None,
        loss_fn="ranking"
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
            True
        )


    def score_hrt(self, triplets):
        """
        Score function is -> h^T * diag(M) * t. We have r = diag(M).

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

        return torch.sum(h * r * t, dim=-1)


    def score_head(self, triplets):
        """
        Get the score for a given set of triplets against *all possible* heads.
        
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

        return torch.sum(h[None, :, :] * r[:, None, :] * t[:, None, :], dim=-1)


    def score_tail(self, triplets):
        """
        Get the score for a given set of triplets against *all possible* tails.

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

        return torch.sum(h[:, None, :] * r[:, None, :] * t[None, :, :], dim=-1)
