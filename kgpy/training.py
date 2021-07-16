import os
import sys
import copy
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

from kgpy.load_data import TrainDataset, TestDataset
from kgpy.evaluation import evaluate_model
from kgpy import utils
from kgpy import sampling


TENSORBOARD_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "runs")


class Trainer:
    """
    Control training of model on a particular dataset
    """

    def __init__(
        self, 
        model, 
        optimizer, 
        data, 
        checkpoint_dir, 
        tensorboard=True
    ):
        self.data = data 
        self.model = model
        self.optimizer = optimizer
        self.tensorboard = tensorboard
        self.device = model._cur_device()
        self.checkpoint_dir = checkpoint_dir

        if tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, model.name, data.dataset_name), flush_secs=3)


    def fit(
            self, 
            epochs, 
            train_batch_size, 
            train_method=None,
            validate_every=5, 
            non_train_batch_size=16, 
            early_stopping=5, 
            save_every=25,
            log_every_n_steps=25,
            negative_samples=1,
            evaluation_method="filtered"
        ):
        """
        Train and validate the model

        Parameters:
        -----------
            epochs: int
                Number of epochs to train for
            train_batch_size: int
                Batch size to use for training
            train_method: str
                None or 1-N
            validate_every: int
                Validate every "n" epochs. Defaults to 5
            non_train_batch_size: int 
                Batch size for non-training data. Defaults to 16
            early_stopping: int
                Stop training if the mean rank hasn't improved in last "n" validation scores. Defaults to 5
            save_every: int
                Save model every "n" epochs. Defaults to 25
            log_every_n_steps: int 
                Log training loss to tensorboard every "n" steps. Defaults to 25
            negative_samples: int
                Number of negative samples to generate for each training sample. Defaults to 1
            evaluation_method: str
                How to evaluate data. Filtered vs raw. Defaults to filtered

        Returns:
        --------
            None
        """
        step = 1
        val_mrr = []
        sampler = self._get_sampler(train_method, train_batch_size, negative_samples)

        for epoch in range(1, epochs+1):

            self.model.train()   
            
            prog_bar = tqdm(sampler, file=sys.stdout)
            prog_bar.set_description(f"Epoch {epoch}")

            for batch in prog_bar:
                step += 1
                batch_loss = self._train_batch(batch, train_method)

                if step % log_every_n_steps == 0 and self.tensorboard:
                    self.writer.add_scalar(f'training_loss', batch_loss, global_step=step)

            if epoch % validate_every == 0:
                val_mrr.append(self._validate_model(epoch, non_train_batch_size, evaluation_method))
    
                # Start checking after accumulate more than val_mrr
                if len(val_mrr) >= early_stopping and np.argmax(val_mrr[-early_stopping:]) == 0:
                    print(f"Validation loss hasn't improved in the last {early_stopping} validation mean rank scores. Stopping training now!", flush=True)
                    break

                # Only save when we know the model performs better
                utils.save_model(self.model, self.optimizer, epoch, step, self.data.dataset_name, self.checkpoint_dir)

            if epoch % save_every == 0:
                utils.save_model(self.model, self.optimizer, epoch, step, self.data.dataset_name, self.checkpoint_dir, suffix=f"epoch_{epoch}")
            
            sampler.reset()



    def _train_batch(self, batch, train_method):
        """
        Train model on single batch

        Parameters:
        -----------
            batch: tuple
                Tuple of head, relations, and tails for each sample in batch
            train_method: str
                Either 1-N or None
        
        Returns:
        -------
        float
            batch loss
        """
        self.optimizer.zero_grad()

        if train_method.upper() == "1-K":
            batch_loss = self._train_batch_1_to_k(batch)
        elif train_method.upper() == "1-N":
            batch_loss = self._train_batch_1_to_n(batch)

        batch_loss = batch_loss.mean()
        batch_loss.backward()

        self.optimizer.step()

        return batch_loss.item()


    def _train_batch_1_to_k(self, batch): 
        """
        Train model on single batch using 1-K training method

        Parameters:
        -----------
            batch: tuple of tuples
                First tuple is positive samples and the second negative. Each ontains head, relations, and tails.

        Returns:
        -------
        loss
            batch loss
        """
        pos_trips, neg_trips = batch[0], batch[1]
        pos_heads, pos_relations, pos_tails = pos_trips[:, 0].to(self.device), pos_trips[:, 1].to(self.device), pos_trips[:, 2].to(self.device)
        neg_heads, neg_relations, neg_tails = neg_trips[:, 0].to(self.device), neg_trips[:, 1].to(self.device), neg_trips[:, 2].to(self.device)

        pos_triplets = torch.stack((pos_heads, pos_relations, pos_tails), dim=1)
        neg_triplets = torch.stack((neg_heads, neg_relations, neg_tails), dim=1)

        pos_scores = self.model(pos_triplets)
        neg_scores = self.model(neg_triplets)

        return self.model.loss(positive_scores=pos_scores, negative_scores=neg_scores)


    def _train_batch_1_to_n(self, batch): 
        """
        Train model on single batch

        Parameters:
        -----------
            batch: tuple of tuples
                First tuple is positive samples and the second negative. Each ontains head, relations, and tails.

        Returns:
        -------
        loss
            batch loss
        """
        if 'bce' not in self.model.loss_fn.__class__.__name__.lower():
            raise ValueError("1-N training can only be used with BCE loss!")

        trips, lbls, trip_type = batch[0], batch[1], batch[2]

        head_trips = trips[trip_type == "head"]
        tail_trips = trips[trip_type == "tail"]

        head_lbls = lbls[trip_type == "head"]
        tail_lbls = lbls[trip_type == "tail"]

        head_scores = self.model(head_trips, mode="head")
        tail_scores = self.model(tail_trips, mode="tail")
        
        all_scores = torch.flatten(torch.cat((head_scores, tail_scores)))
        all_lbls = torch.flatten(torch.cat((head_lbls, tail_lbls)))

        return self.model.loss(all_scores=all_scores, all_targets=all_lbls)


       
    def _validate_model(self, epoch, batch_size, evaluation_method):
        """
        Evaluate model on val set

        Parameters:
        -----------
            epoch: int
                epoch number
            batch_size: int
                size of batch
            evaluation method:
                Filtered or raw

        Returns:
        --------
        float
            mean reciprocal rank
        """
        dataloader = torch.utils.data.DataLoader(
                        TestDataset(self.data.dataset_name, self.data.valid_triplets, self.data.all_triplets, self.data.num_entities, evaluation_method), 
                        batch_size=batch_size,
                        num_workers=8
                    )

        mr, mrr, hits_at_1, hits_at_3, hits_at_10 = evaluate_model(self.model, dataloader)

        if self.tensorboard:
            self.writer.add_scalar('Hits@1%' , hits_at_1, epoch)
            self.writer.add_scalar('Hits@3%' , hits_at_3, epoch)
            self.writer.add_scalar('Hits@10%', hits_at_10, epoch)
            self.writer.add_scalar('MR'      , mr, epoch)
            self.writer.add_scalar('MRR'     , mrr, epoch)

        return mrr



    def _get_sampler(self, train_method, bs, num_negative=None):
        """
        Retrieve a sampler object for the type of train method
        """
        train_method = train_method.upper()

        if train_method == "1-K":
            sampler = sampling.One_to_K(
                        self.data['train'], 
                        bs, 
                        self.data.num_entities, 
                        self.device,
                        num_negative=num_negative
                    )
        elif train_method == "1-N":
            sampler = sampling.One_to_N(
                        self.data['train'], 
                        bs, 
                        self.data.num_entities, 
                        self.device
                    )
        else:
            raise ValueError(f"Invalid train method `{train_method}`.")
        
        return sampler




