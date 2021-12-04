"""
Implementation of ConvE. 

See paper for more details - https://arxiv.org/abs/1707.01476.
"""
import torch
import torch.nn.functional as F

from .base.single_emb_model import SingleEmbeddingModel


class ConvE(SingleEmbeddingModel):
    def __init__(self, 
        num_entities, 
        num_relations, 
        emb_dim=200, 
        filters=32,
        ker_sz=3,
        k_h=20,
        # Code itself is hidden_drop=.3 but he mentions a higher regularization rate (0.5) for FB15K-237 here
        # https://github.com/TimDettmers/ConvE/issues/52#issuecomment-537231786
        hidden_drop=.5,
        input_drop=.2,
        feat_drop=.2,
        margin=1, 
        regularization='l2',
        reg_weight=0,
        weight_init=None,
        loss_fn="bce",
        device='cuda'
    ):
        super().__init__(
            type(self).__name__,
            num_entities, 
            num_relations, 
            emb_dim, 
            emb_dim,  
            margin, 
            regularization,
            reg_weight,
            weight_init, 
            loss_fn,
            True,
            device
        )        
        self.inp_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feat_drop)

        # emb_dim = kernel_h * kernel_w
        self.k_h = k_h
        self.k_w = emb_dim // k_h
        self.filters = filters
        self.ker_sz = ker_sz

        # TODO: Determine why wrong
        # flat_sz_h = int(2*self.k_w) - self.ker_sz + 1
        # flat_sz_w = self.k_h - self.ker_sz + 1
        # self.hidden_size = flat_sz_h*flat_sz_w*filters
        self.hidden_size = 9728

        self.conv1 = torch.nn.Conv2d(1, filters, kernel_size=(ker_sz, ker_sz), stride=1, padding=0)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(emb_dim)

        self.register_parameter('b', torch.nn.Parameter(torch.zeros(num_entities)))

        self.fc = torch.nn.Linear(self.hidden_size, emb_dim)


    def score_function(self, e1, rel):
        """
        Scoring process of triplets

        Parameters:
        -----------
            e1: torch.Tensor
                entities passed through ConvE
            e2: torch.Tensor
                entities scored against for link prediction
            rel: torch.Tensor
                relaitons passed through ConvE
        
        Returns:
        --------
        torch.Tensor
            Raw scores to be multipled by entities (e.g. dot product)
        """
        triplets = torch.cat([e1, rel], 2)

        stacked_inputs = self.bn0(triplets)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x


    def score_hrt(self, triplets):
        """
        Pass through ConvE.

        Note: Only work for 1-N

        Parameters:
        -----------
            triplets: list
                List of triplets of form (sub, rel, obj)

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        e1_embedded  = self.ent_embs(triplets[:, 0]).view(-1, 1, self.k_h, self.k_w)
        rel_embedded = self.rel_embs(triplets[:, 1]).view(-1, 1, self.k_h, self.k_w)

        # Each must only be multiplied by entity belong to *own* triplet!!!
        e2_embedded  = self.ent_embs(triplets[:, 2])

        x = self.score_function(e1_embedded, rel_embedded)

        # Again, they should should only multiply with own entities
        # This is the diagonal of the matrix product in 1-N
        x = (x * e2_embedded).sum(dim=1).reshape(-1, 1)

        # TODO: ???
        # x += self.b.expand_as(x)

        return x


    def score_head(self, triplets):
        """
        Get the score for a given set of triplets against *all possible* heads.
        
        Parameters:
        -----------
            triplets: list
                List of triplets of form (rel, obj)

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        e1_embedded  = self.ent_embs(triplets[:, 1]).view(-1, 1, self.k_h, self.k_w)
        rel_embedded = self.rel_embs(triplets[:, 0]).view(-1, 1, self.k_h, self.k_w)

        x = self.score_function(e1_embedded, rel_embedded)
        x = torch.mm(x, self.ent_embs.weight.transpose(1,0))
        x += self.b.expand_as(x)

        return x

        
    # TODO: For now just pass to score_head since same
    def score_tail(self, triplets):
        """
        Get the score for a given set of triplets against *all possible* tails.

        Parameters:
        -----------
            triplets: list
                List of triplets of form (rel, sub)

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        return self.score_head(triplets)


    def sym_regularization(self, triplets):
        """
        Symmetric regularization.
        """
        # rel_embedded for inverse
        # when greater than self.num_relations subract otherwise add        
        rel_ix = triplets[:, 1]
        num_non_inv_rels = int(self.num_relations / 2)
        inv_rel_ix = torch.where(rel_ix >= num_non_inv_rels, rel_ix - num_non_inv_rels, rel_ix + num_non_inv_rels)

        inv_triplets = torch.cat((triplets[:, 2].unsqueeze(1), inv_rel_ix.unsqueeze(1), triplets[:, 0].unsqueeze(1)), dim=1).long().to(self.device)

        tail_scores = self.score_hrt(triplets.long())
        head_scores = self.score_hrt(inv_triplets)

        return torch.linalg.norm(head_scores - tail_scores, dim=1, ord=2).sum()
        


    def score_both(self, triplets):
        """
        for (h, r):
            for all t \in E:
                f(h, r, t) + f(t, r^-1, h)
        """
        e1_embedded  = self.ent_embs(triplets[:, 0]).view(-1, 1, self.k_h, self.k_w)
        rel_embedded = self.rel_embs(triplets[:, 1]).view(-1, 1, self.k_h, self.k_w)
        # e2_embedded  = self.ent_embs(triplets[:, 2]).view(-1, 1, self.k_h, self.k_w)

        # rel_embedded for inverse
        # when greater than self.num_relations subract otherwise add        
        rel_ix = triplets[:, 1]
        num_non_inv_rels = int(self.num_relations / 2)
        inv_rel_ix = torch.where(rel_ix >= num_non_inv_rels, rel_ix - num_non_inv_rels, rel_ix + num_non_inv_rels)
        # inv_rel_embedded = self.rel_embs(inv_rel_ix).view(-1, 1, self.k_h, self.k_w)

        """
        TODO: Incorrect!!!

        1. Need to construct (h, r, t) for all t \in V
        2. For each sample created in 1, construct (t, r^-1, h)
        3. Scores
        4. Combine
        """

        # Tail Scores
        # tail_out = (128, 14541)
        tail_out = self.score_function(e1_embedded, rel_embedded)
        tail_out = torch.mm(tail_out, self.ent_embs.weight.transpose(1,0))
        tail_out += self.b.expand_as(tail_out)


        inv_heads = torch.Tensor(range(self.num_entities)).repeat(triplets.shape[0]).long().to(self.device)
        inv_rels = inv_rel_ix.repeat_interleave(self.num_entities)
        inv_tails = triplets[:, 0].repeat_interleave(self.num_entities)


        print(self.ent_embs.weight.device)
        print(inv_heads.device)
        # print(inv_rel_ix.shape, triplets[:, 0].shape, inv_rels.shape, inv_tails.shape)
        # exit()

        e2_embedded  = self.ent_embs(inv_heads).view(-1, 1, self.k_h, self.k_w)
        inv_rel_embedded = self.rel_embs(inv_rels).view(-1, 1, self.k_h, self.k_w)

        # Head Scores
        head_out = self.score_function(e2_embedded, inv_rel_embedded)

        print(head_out.shape)

        # head_out = torch.mm(head_out, self.ent_embs.weight.transpose(1,0))
        # head_out += self.b.expand_as(head_out)



        return .5 * tail_out + .5 * head_out

