import torch
import numpy as np

if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


def hit_at_k(scores, true_entities, k):
    """
    Check # of times true answer is in top k scores for a batch

    Args:
        scores: Scores predicting the link for each entity for our samples
        true_entities: True entities corresponding to each sample
        k: Top-k to check

    Return:
        Number of times the true answer was in the top k
    """
    zero_tensor = torch.tensor([0], device=device)
    one_tensor = torch.tensor([1], device=device)
    _, topk_ix = scores.topk(k=k, largest=True)

    return torch.where(topk_ix == true_entities[:, torch.arange(k)], one_tensor, zero_tensor).sum().item()


def mean_rank(scores, true_entities):
    """
    The mean rank of the true link among the scores.

    Lower score function == better

    Args:
        scores: Scores predicting the link for each entity for our samples
        true_entities: True entities corresponding to each sample

    Returns:
        Mean rank of correct link scores
    """
    ranked_scores = scores.argsort(descending=True)
    rank_true_entities = torch.where(ranked_scores == true_entities)

    return rank_true_entities[1].float().mean().item()



def mean_reciprocal_rank(scores, true_entities):
    """
    The mean reciprocal rank of the true link among the scores

    Args:
        scores: Scores predicting the link for each entity for our samples
        true_entities: True entities corresponding to each sample

    Returns:
        Mean reciprocal rank of correct link scores
    """
    ranked_scores = scores.argsort(descending=True)
    rank_true_entities = torch.where(ranked_scores == true_entities)
    
    # Add constant to denominator for div/0
    rank_true_entities_no_zero = torch.Tensor([i if i != 0 else torch.tensor(1) for i in rank_true_entities[1]])

    return torch.mean(1.0 / rank_true_entities_no_zero).item()


def evaluate_model(model, data, all_dataset):
    """
    Evaluate the model on a given set of data. Return the relevant metrics

    Args:
        model: nn.Module object
        data: DataLoader object for validation or testing set
        all_dataset: AllDataSet object

    Returns:
        Tuple of (mean-rank, mean-recipocal-rank, hits_at_1, hits_at_3, hits_at_10)
    """
    num_batches = 0
    total_samples = 0
    mr, mrr = 0, 0
    hits_at_1, hits_at_3, hits_at_10 = 0, 0, 0

    # To ensure no updates
    model.eval()

    # Every entity id
    entity_ids = torch.arange(end=all_dataset.num_entities, device=device)

    mean_recip_ranks = []


    for batch in data:
        batch_size = batch[0].size()[0]
        batch_heads, batch_relations, batch_tails = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        # Repeat all entities for number of samples in batch
        all_entities = entity_ids.repeat(batch_size, 1)

        # Repeat each head in our batch nall_entities.size times
        # e.g. [[4, 4, ..., 4], [1, 1, ..., 1]]
        batch_heads_repeat = batch_heads.reshape(-1, 1).repeat(1, all_entities.size()[1])
        batch_relations_repeat = batch_relations.reshape(-1, 1).repeat(1, all_entities.size()[1])
        batch_tails_repeat = batch_tails.reshape(-1, 1).repeat(1, all_entities.size()[1])

        with torch.no_grad():
            head_triplets = torch.stack((all_entities, batch_relations_repeat, batch_tails_repeat), dim=2).reshape(-1, 3)
            tail_triplets = torch.stack((batch_heads_repeat, batch_relations_repeat, all_entities), dim=2).reshape(-1, 3)

            # Calc scores
            # Reshapes so each entry contains score for the ith triplet
            head_scores = model.score_function(head_triplets).reshape(batch_size, -1)
            tails_scores = model.score_function(tail_triplets).reshape(batch_size, -1)

            # Calc scores for original batch that were removed
            # batch_triplets = torch.stack((batch_heads, batch_relations, batch_tails), dim=1).reshape(-1, 3)
            # batch_scores = model.score_function(batch_triplets).reshape(batch_size, -1)


        # Combine to evaluate in one time
        all_scores = torch.cat((head_scores, tails_scores), dim=0)            
        true_entities = torch.cat((batch_heads_repeat, batch_tails_repeat))

        ################
        # Calc metrics
        ################
        num_batches += 1
        total_samples += batch_size

        # Mean Ranks
        mr += mean_rank(all_scores, true_entities)
        mrr += mean_reciprocal_rank(all_scores, true_entities)

        # Hits @ k
        hits_at_1 += hit_at_k(all_scores, true_entities, 1)
        hits_at_3 += hit_at_k(all_scores, true_entities, 3)
        hits_at_10 += hit_at_k(all_scores, true_entities, 10)


    # Divide by number of batches for those two since they are a mean
    mr = mr / num_batches  
    mrr = mrr / num_batches * 100
    hits_at_1 = hits_at_1 / total_samples * 100
    hits_at_3 = hits_at_3 / total_samples * 100
    hits_at_10 = hits_at_10 / total_samples * 100

    print(f"\nMR: {mr} \nMRR: {mrr} \nHits@1%: {hits_at_1} \nHits@3%: {hits_at_3} \nHits@10%: {hits_at_10}\n")

    return mr, mrr, hits_at_1, hits_at_3, hits_at_10


def test_model(model, data, batch_size):
    """
    Test the model. Wrapper for evaluate_model

    Args:
        model: 
            pytorch model
        data:
            AllDataset object
        batch_size: 
            Batch for loader object.

    Returns:
        Tuple of (mean-rank, mean-recipocal-rank, hits_at_1, hits_at_3, hits_at_10)
    """
    dataloader = torch.utils.data.DataLoader(self.data['test'], batch_size=batch_size)

    return evaluate_model(self.model, dataloader, self.data)