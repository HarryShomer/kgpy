import sys
import torch
import numpy as np
from tqdm import tqdm

from kgpy.load_data import TestDataset



class Evaluation:

    def __init__(self, split, all_data, inverse, eval_method='filtered', bs=128, device='cpu'):
        self.bs = bs
        self.split = split
        self.data = all_data
        self.device = device
        self.inverse = inverse
        self.eval_method = eval_method

        if self.eval_method != "filtered":
            raise NotImplementedError("TODO: Implement raw evaluation metrics")


    @property
    def triplets(self):
        return self.data[self.split]



    def evaluate(self, model):
        """
        Evaluate the model on the valid/test set

        Parameters:
        -----------
            model: EmbeddingModel
                model we are fitting

        Returns:
        --------
        dict
            eval metrics
        """
        hits_k_vals = [1, 3, 10]
        metrics = ["samples", "mr", "mrr", "hits@1", "hits@3", "hits@10"]
        results = {m : 0 for m in metrics}

        dataloader = torch.utils.data.DataLoader(
                        TestDataset(self.triplets, self.data.all_triplets, self.data.num_entities, inverse=self.inverse, device=self.device), 
                        batch_size=self.bs,
                        num_workers=8
                    )

        model.eval()
        with torch.no_grad():

            steps = 0
            prog_bar = tqdm(dataloader, file=sys.stdout)
            prog_bar.set_description(f"Evaluating model")

            for batch in prog_bar:
                
                steps += 1
                trips, obj, lbls = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)

                if not self.inverse:
                    raise NotImplemented("TODO: Evaluation when not adding inverse triplets")
                    # head_trips = trips[trip_type == "head"]
                    # tail_trips = trips[trip_type == "tail"]

                    # head_lbls = lbls[trip_type == "head"]
                    # tail_lbls = lbls[trip_type == "tail"]

                    # head_lbls = self.index(head_trips[:, :2])
                    # tail_lbls = self.index(tail_trips[:, :2])

                    # head_scores = model(trips, mode="head")
                    # tail_scores = model(trips, mode="tail")
                
                    # preds = torch.flatten(torch.cat((head_scores, tail_scores)))
                    # lbls = torch.flatten(torch.cat((head_lbls, tail_lbls)))
                else:
                    preds = model(trips, mode="tail")
                
                # [0, 1, 2, ..., BS]
                b_range	= torch.arange(preds.size()[0], device=self.device)

                # Extract scores for correct object for each sample in batch
                target_pred	= preds[b_range, obj]

                # self.byte() is equivalent to self.to(torch.uint8)
                # This makes true triplets not in the batch equal to -1000000 by first setting **all** true triplets to -1000000
                # and then pluggin the original preds for the batch samples back in
                preds 			    = torch.where(lbls.byte(), -torch.ones_like(preds) * 10000000, preds)
                preds[b_range, obj] = target_pred

                # Holds rank of correct score (e.g. When 1 that means the score for the correct triplet had the highest score)
                ranks = 1 + torch.argsort(torch.argsort(preds, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                ranks = ranks.float()

                results['samples'] += torch.numel(ranks) 
                results['mr']      += torch.mean(ranks).item() 
                results['mrr']     += torch.mean(1.0/ranks).item()

                for k in hits_k_vals:
                    results[f'hits@{k}'] += torch.numel(ranks[ranks <= k])

        ### Average out results
        results['mr']  = results['mr']  / steps 
        results['mrr'] = results['mrr'] / steps * 100

        for k in hits_k_vals:
            results[f'hits@{k}'] = results[f'hits@{k}'] / results['samples'] * 100

        return results


