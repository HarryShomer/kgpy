import torch
import numpy as np

from load_data import TestDataset

if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


def mean_rank(batch_scores):
    """
    The mean rank of the true link among the scores.

    The true score is in the zeroth element!!!

    Args:
        batch_scores: list of Tensors
            Scores predicting the link for each entity for our samples

    Returns:
        Mean rank of correct link scores
    """
    ranks = []

    for scores in batch_scores:
        ranked_scores = scores.argsort(descending=True)
        
        # True entity is always placed as first index in scores
        true_entity = torch.Tensor([0]).repeat(scores.size()[0]).to(device)
        rank_true_entities = torch.where(ranked_scores == true_entity)

        ranks.append(rank_true_entities[0].item() + 1)  # since 0 indexed

    return ranks


def mean_reciprocal_rank(batch_scores):
    """
    The mean reciprocal rank of the true link among the scores

    The true score is in the zeroth element!!!

    Args:
        batch_scores: list of Tensors
            Scores predicting the link for each entity for our samples

    Returns:
        Mean reciprocal rank of correct link scores
    """
    reciprocal_ranks = []

    for scores in batch_scores:
        ranked_scores = scores.argsort(descending=True)
        
        # True entity is always placed as first index in scores
        true_entity = torch.Tensor([0]).repeat(scores.size()[0]).to(device)
        rank_true_entities = torch.where(ranked_scores == true_entity)

        # + 1 since 0 indexed so if get #1 hit would be index 0 and div/0
        reciprocal_ranks.append(1 / (rank_true_entities[0].item() + 1))


    return reciprocal_ranks



def hit_at_k(batch_scores, k):
    """
    Check # of times true answer is in top k scores for a batch

    Args:
        batch_scores: list of Tensors
            Scores predicting the link for each entity for our samples
        k: 
            Top-k to check

    Return:
        Number of times the true answer was in the top k
    """
    zero_tensor = torch.tensor([0], device=device)
    one_tensor = torch.tensor([1], device=device)
    true_entity = torch.Tensor([0]).repeat(k).to(device)   # True entity is always placed as first index in scores

    hits = 0

    for scores in batch_scores:
        _, topk_ix = scores.topk(k=k, largest=True)        
        hits += torch.where(topk_ix == true_entity, one_tensor, zero_tensor).sum().item()

    return hits



def evaluate_model(model, data):
    """
    Evaluate the model on a given set of data. Return the relevant metrics

    Args:
        model: nn.Module object
        data: DataLoader object for validation or testing set

    Returns:
        Tuple of (mean-rank, mean-recipocal-rank, hits_at_1, hits_at_3, hits_at_10)
    """
    num_batches = 0
    total_samples = 0
    mr, mrr = [], []
    hits_at_1, hits_at_3, hits_at_10 = 0, 0, 0

    # To ensure no updates
    model.eval()

    with torch.no_grad():
        for nn, batch in enumerate(data):
            batch_size = batch[0].size()[0]

            true_triplets = batch[0].to(device)
            corrupted_head_triplets = batch[1].to(device)
            corrupted_tail_triplets = batch[2].to(device)

            # Holds the number of corrupted <Head/Tail> triplets for the ith true triplet in the batch
            # Needed since filtering below fucks with dimensions
            num_corrupt_head = torch.sum(corrupted_head_triplets[:,:,3] != 0, dim=1).tolist()
            num_corrupt_tail = torch.sum(corrupted_tail_triplets[:,:,3] != 0, dim=1).tolist()

            # Filter out true triplets
            # Can then get rid of indicator (4th dim in triplet)
            corrupted_head_triplets = corrupted_head_triplets[corrupted_head_triplets[:,:,3] != 0][:, :3]
            corrupted_tail_triplets = corrupted_tail_triplets[corrupted_tail_triplets[:,:,3] != 0][:, :3]

            head_scores = model.score_function(corrupted_head_triplets)
            tail_scores = model.score_function(corrupted_tail_triplets)
            true_scores = model.score_function(true_triplets).reshape(batch_size, -1)

            # Head and tail list are now (-1, 3) so we split into chunks based on number of <Head/Tail> corrupted triplets
            head_scores = torch.split(head_scores, num_corrupt_head)
            tail_scores = torch.split(tail_scores, num_corrupt_tail)

            # Free space
            del batch, corrupted_head_triplets, corrupted_tail_triplets

            # Cat all scores for a given true triplet
            # True entity is in the zeroth element!!!
            all_scores = []
            for i in range(true_scores.size()[0]):
                all_scores.append(torch.cat((true_scores[i], head_scores[i], tail_scores[i]), dim=0))

            # Free space pt 2
            del head_scores, tail_scores, true_scores

            ################
            # Calc metrics
            ################
            total_samples += batch_size

            # Mean Ranks
            mr.extend(mean_rank(all_scores))
            mrr.extend(mean_reciprocal_rank(all_scores))

            # Hits @ k
            hits_at_1 += hit_at_k(all_scores, 1)
            hits_at_3 += hit_at_k(all_scores, 3)
            hits_at_10 += hit_at_k(all_scores, 10)


    mr =  round(np.mean(mr), 2)       
    mrr = round(np.mean(mrr) * 100, 2)
    hits_at_1 = round(hits_at_1 / total_samples * 100, 2) 
    hits_at_3 = round(hits_at_3 / total_samples * 100, 2) 
    hits_at_10 = round(hits_at_10 / total_samples * 100, 2) 

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
    dataloader = torch.utils.data.DataLoader(
                    TestDataset(data.dataset_name, data.test_triplets, data.all_triplets, data.num_entities), 
                    batch_size=batch_size,
                    num_workers=8
                )

    return evaluate_model(model, dataloader)