"""
Implementation of Complex. 

See paper for more details - http://proceedings.mlr.press/v48/trouillon16.pdf.
"""
import torch

from .base_emb_model import ComplexEmbeddingModel


class ComplEx(ComplexEmbeddingModel):

    def __init__(
        self, 
        num_entities, 
        num_relations, 
        emb_dim=100, 
        margin=1, 
        regularization = 'l2',
        reg_weight = 1e-6,
        weight_init="normal",
        loss_fn= "ranking"  #"softplus"
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
            True,
        )


    def score_hrt(self, triplets):
        """        
        Score =  <Re(h), Re(r), Re(t)>
               + <Im(h), Re(r), Im(t)>
               + <Re(h), Im(r), Im(t)>
               - <Im(h), Im(r), Re(t)>

        Parameters:
        -----------
            triplets: list
                List of triplets

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        h_re = self.entity_emb_re(triplets[:, 0])
        h_im = self.entity_emb_im(triplets[:, 0])
        t_re = self.entity_emb_re(triplets[:, 2])
        t_im = self.entity_emb_im(triplets[:, 2])
        r_re = self.relation_emb_re(triplets[:, 1])
        r_im = self.relation_emb_im(triplets[:, 1])

        return torch.sum(
                  (h_re * r_re * t_re) 
                + (h_im * r_re * t_im)
                + (h_re * r_im * t_im)
                - (h_im * r_im * t_re)
                , dim=-1
            ) 


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
        h_re = self.entity_emb_re(torch.arange(self.num_entities, device=self._cur_device()).long())
        h_im = self.entity_emb_im(torch.arange(self.num_entities, device=self._cur_device()).long())
        t_re = self.entity_emb_re(triplets[:, 1])
        t_im = self.entity_emb_im(triplets[:, 1])
        r_re = self.relation_emb_re(triplets[:, 0])
        r_im = self.relation_emb_im(triplets[:, 0])

        return torch.sum(
                  (h_re[None, :, :] * r_re[:, None, :] * t_re[:, None, :]) 
                + (h_im[None, :, :] * r_re[:, None, :] * t_im[:, None, :])
                + (h_re[None, :, :] * r_im[:, None, :] * t_im[:, None, :])
                - (h_im[None, :, :] * r_im[:, None, :] * t_re[:, None, :])
                , dim=-1
            ) 


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
        t_re = self.entity_emb_re(torch.arange(self.num_entities, device=self._cur_device()).long())
        t_im = self.entity_emb_im(torch.arange(self.num_entities, device=self._cur_device()).long())
        h_re = self.entity_emb_re(triplets[:, 1])
        h_im = self.entity_emb_im(triplets[:, 1])
        r_re = self.relation_emb_re(triplets[:, 0])
        r_im = self.relation_emb_im(triplets[:, 0])

        return torch.sum(
                  (h_re[:, None, :] * r_re[:, None, :] * t_re[None, :, :]) 
                + (h_im[:, None, :] * r_re[:, None, :] * t_im[None, :, :])
                + (h_re[:, None, :] * r_im[:, None, :] * t_im[None, :, :])
                - (h_im[:, None, :] * r_im[:, None, :] * t_re[None, :, :])
                , dim=-1
            ) 
