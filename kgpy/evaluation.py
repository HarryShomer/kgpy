import torch

if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


def hit_at_k(scores, true_entities, k):
    """
    Check # of times true answer is in top k scores

    Args:
        scores: Scores predicting the link for each entity for our samples
        true_entities: True entities corresponding to each sample
        k: Top-k to check

    Return:
        Number of times the true answer was in the top k
    """
    zero_tensor = torch.tensor([0], device=device)
    one_tensor = torch.tensor([1], device=device)
    _, topk_ix = scores.topk(k=k, largest=False)

    return torch.where(topk_ix == true_entities[:, torch.arange(k)], one_tensor, zero_tensor).sum().item()


def mean_rank(scores, true_entities):
    """
    The mean rank of the true link among the scores

    Args:
        scores: Scores predicting the link for each entity for our samples
        true_entities: True entities corresponding to each sample

    Returns:
        Mean rank of correct link scores
    """
    ranked_scores = scores.argsort(descending=False)
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
    ranked_scores = scores.argsort(descending=False)
    rank_true_entities = torch.where(ranked_scores == true_entities)

    return (1.0 / rank_true_entities[1]).sum().item() / rank_true_entities[0].size()[0]



def test_model(model, data, num_entities):
    """
    Test the model on a given set of data. Return the relevant metrics

    Args:
        model: nn.Module object
        data: DataLoader object for validation or testing set
        num_entities: # of unique entities in our data set

    Returns:
        Tuple of (mean-rank, mean-recipocal-rank, hits_at_1, hits_at_3, hits_at_10)
    """
    model.eval()

    # Every entity id
    entity_ids = torch.arange(end=num_entities, device=device)

    total_samples = 0
    mr, mrr = 0, 0
    hits_at_1, hits_at_3, hits_at_10 = 0, 0, 0

    for batch in data:
        batch_size = batch[0].size()[0]
        batch_heads, batch_relations, batch_tails = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        # Repeat all entities for number of samples in batch
        all_entities = entity_ids.repeat(batch_size, 1)

        # Repeat each head in our batch nall_entities.size times
        # e.g. [[4, 4, ..., 4], [1, 1, ..., 1]]
        batch_heads = batch_heads.reshape(-1, 1).repeat(1, all_entities.size()[1])
        batch_relations = batch_relations.reshape(-1, 1).repeat(1, all_entities.size()[1])
        batch_tails = batch_tails.reshape(-1, 1).repeat(1, all_entities.size()[1])

        with torch.no_grad():
            head_triplets = torch.stack((all_entities, batch_relations, batch_tails), dim=2).reshape(-1, 3)
            tail_triplets = torch.stack((batch_heads, batch_relations, all_entities), dim=2).reshape(-1, 3)
            
            # Calc scores
            # Reshapes so each entry contains score for the ith triplet
            head_scores = model.score_function(head_triplets).reshape(batch_size, -1)
            tails_scores = model.score_function(tail_triplets).reshape(batch_size, -1)

        # Combine to evaluate in one time
        all_scores = torch.cat((head_scores, tails_scores), dim=0)            
        true_entities = torch.cat((batch_heads, batch_tails))

        ##################################
        # Weight each metric by batch size
        ##################################
        total_samples += batch_size

        # Mean Ranks
        mr += mean_rank(all_scores, true_entities) * batch_size
        mrr += mean_reciprocal_rank(all_scores, true_entities) * batch_size

        # Hits @ k
        hits_at_1 += hit_at_k(all_scores, true_entities, 1) * batch_size
        hits_at_3 += hit_at_k(all_scores, true_entities, 3) * batch_size
        hits_at_10 += hit_at_k(all_scores, true_entities, 10) * batch_size


    return mr / total_samples, mrr / total_samples, hits_at_1 / total_samples, hits_at_3 / total_samples, hits_at_10 / total_samples
