import sys
import torch
import numpy as np
from tqdm import tqdm

from kgpy.datasets import TestDataset



class Evaluation:

    def __init__(self, split, all_data, inverse, eval_method='filtered', bs=128, device='cpu'):
        self.bs = bs
        self.split = split
        self.data = all_data
        self.device = device
        self.inverse = inverse
        self.eval_method = eval_method
        self.hits_k_vals = [1, 3, 10]

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
                tail_trips, obj, tail_lbls = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)

                if not self.inverse:
                    head_trips, sub, head_lbls = batch[3].to(self.device), batch[4].to(self.device), batch[5].to(self.device)

                    head_preds = model(head_trips, mode="head")
                    tail_preds = model(tail_trips, mode="tail")
                
                    self.calc_metrics(head_preds, sub, head_lbls, results)
                    self.calc_metrics(tail_preds, obj, tail_lbls, results)
                else:
                    preds = model(tail_trips, mode="tail")
                    self.calc_metrics(preds, obj, tail_lbls, results)

        ### Average out results
        results['mr']  = results['mr']  / steps 
        results['mrr'] = results['mrr'] / steps * 100

        for k in self.hits_k_vals:
            results[f'hits@{k}'] = results[f'hits@{k}'] / results['samples'] * 100

        return results



    def calc_metrics(self, preds, ent, lbls, results):
        """
        Calculate the metrics for a number of samples.

        `results` dict is modified inplace

        Parameters:
        -----------
            preds: Tensor
                Score for triplets
            ent: Tensor
                Correct index for missing entity for triplet
            lbls: Tensor
                Tensor holding whether triplet tested is true or not
            results: dict
                Holds results for eval run so far
        
        Returns:
        --------
        None
        """
        # [0, 1, 2, ..., BS]
        b_range	= torch.arange(preds.size()[0], device=self.device)

        # Extract scores for correct object for each sample in batch
        target_pred	= preds[b_range, ent]

        # self.byte() is equivalent to self.to(torch.uint8)
        # This makes true triplets not in the batch equal to -1000000 by first setting **all** true triplets to -1000000
        # and then pluggin the original preds for the batch samples back in
        preds = torch.where(lbls.byte(), -torch.ones_like(preds) * 10000000, preds)
        preds[b_range, ent] = target_pred

        # Holds rank of correct score (e.g. When 1 that means the score for the correct triplet had the highest score)
        ranks = 1 + torch.argsort(torch.argsort(preds, dim=1, descending=True), dim=1, descending=False)[b_range, ent]
        ranks = ranks.float()

        results['samples'] += torch.numel(ranks) 
        results['mr']      += torch.mean(ranks).item() 
        results['mrr']     += torch.mean(1.0/ranks).item()

        for k in self.hits_k_vals:
            results[f'hits@{k}'] += torch.numel(ranks[ranks <= k])

