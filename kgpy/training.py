import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from kgpy.evaluation import Evaluation
from kgpy import utils
from kgpy import sampling


TENSORBOARD_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "runs")


from time import time


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
        tensorboard=False
    ):
        self.data = data 
        self.model = model
        self.inverse = data.inverse
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
            train_method,
            validate_every=5, 
            non_train_batch_size=64, 
            early_stopping=5, 
            save_every=25,
            log_every_n_steps=100,
            negative_samples=1,
            eval_method="filtered",
            label_smooth=0
        ):
        """
        Train, validate, and test the model

        Parameters:
        -----------
            epochs: int
                Number of epochs to train for
            train_batch_size: int
                Batch size to use for training
            train_method: str
                1-K or 1-N
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
            eval_method: str
                How to evaluate data. Filtered vs raw. Defaults to filtered
            label_smooth: float
                Label smoothing when training

        Returns:
        --------
            None
        """
        step = 1
        val_mrr = []
        sampler = self._get_sampler(train_method, train_batch_size, negative_samples)
        model_eval = Evaluation("valid", self.data, self.inverse, eval_method=eval_method, bs=non_train_batch_size, device=self.device)

        for epoch in range(1, epochs+1):
            epoch_loss = torch.Tensor([0]).to(self.device)

            prog_bar = tqdm(sampler, file=sys.stdout)
            prog_bar.set_description(f"Epoch {epoch}")
            
            self.model.train()
            
            for batch in prog_bar:
                batch_loss = self._train_batch(batch, train_method, label_smooth)
                
                step += 1
                epoch_loss += batch_loss

                if step % log_every_n_steps == 0 and self.tensorboard:
                    self.writer.add_scalar(f'training_loss', batch_loss.item(), global_step=step)
                
            print(f"Epoch {epoch} loss:", epoch_loss.item())

            if epoch % validate_every == 0:
                val_mrr.append(self._validate_model(model_eval, epoch))

                # Start checking after accumulate more than val_mrr
                if len(val_mrr) >= early_stopping and np.argmax(val_mrr[-early_stopping:]) == 0:
                    print(f"Validation loss hasn't improved in the last {early_stopping} validation mean rank scores. Stopping training now!", flush=True)
                    break

                #TODO: Needed?
                # Only save when we know the model performs better
                utils.save_model(self.model, self.optimizer, epoch, step, self.data.dataset_name, self.checkpoint_dir)

            if epoch % save_every == 0:
                utils.save_model(self.model, self.optimizer, epoch, step, self.data.dataset_name, self.checkpoint_dir, suffix=f"epoch_{epoch}")
            
            sampler.reset()


        self._test_model(eval_method, non_train_batch_size)
   


    def _train_batch(self, batch, train_method, label_smooth):
        """
        Train model on single batch

        Parameters:
        -----------
            batch: tuple
                Tuple of head, relations, and tails for each sample in batch
            train_method: str
                Either 1-N or None
            label_smooth: float
                Label smoothing
        
        Returns:
        -------
        float
            batch loss
        """
        self.optimizer.zero_grad()

        if train_method.upper() == "1-K":
            batch_loss = self._train_batch_1_to_k(batch, label_smooth)
        elif train_method.upper() == "1-N":
            batch_loss = self._train_batch_1_to_n(batch, label_smooth)
 
        batch_loss = batch_loss.mean()
        batch_loss.backward()

        self.optimizer.step()

        return batch_loss


    def _train_batch_1_to_k(self, batch, label_smooth): 
        """
        Train model on single batch using 1-K training method

        Parameters:
        -----------
            batch: tuple of tuples
                First tuple is positive samples and the second negative. Each ontains head, relations, and tails.
            label_smooth: float
                Amount of label smoothing to use

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


    def _train_batch_1_to_n(self, batch, label_smooth): 
        """
        Train model on single batch

        Parameters:
        -----------
            batch: tuple of tuples
                First tuple is positive samples and the second negative. Each ontains head, relations, and tails.
            label_smooth: float
                Amount of label smoothing to use

        Returns:
        -------
        loss
            batch loss
        """
        if 'bce' not in self.model.loss_fn.__class__.__name__.lower():
            raise ValueError("1-N training can only be used with BCE loss!")

        if not self.inverse:
            trips, lbls, trip_type = batch[0], batch[1], batch[2]

            head_trips = trips[trip_type == "head"]
            tail_trips = trips[trip_type == "tail"]

            head_lbls = lbls[trip_type == "head"]
            tail_lbls = lbls[trip_type == "tail"]

            head_scores = self.model(head_trips, mode="head")
            tail_scores = self.model(tail_trips, mode="tail")
        
            all_scores = torch.flatten(torch.cat((head_scores, tail_scores)))
            all_lbls = torch.flatten(torch.cat((head_lbls, tail_lbls)))
        else:
            trips, all_lbls = batch[0], batch[1]
            all_scores = self.model(trips, mode="tail")

        if label_smooth != 0.0:
            all_lbls = (1.0 - label_smooth)*all_lbls + (1.0 / self.data.num_entities)


        return self.model.loss(all_scores=all_scores, all_targets=all_lbls)



    def _validate_model(self, model_eval, epoch):
        """
        Evaluate model on val set

        Parameters:
        -----------
            model_eval: Evaluation
                Evaluation object
            epoch: int
                epoch number

        Returns:
        --------
        float
            mean reciprocal rank
        """
        results = model_eval.evaluate(self.model)

        if self.tensorboard:
            self.writer.add_scalar('Hits@1%' , results['hits@1'], epoch)
            self.writer.add_scalar('Hits@3%' , results['hits@3'], epoch)
            self.writer.add_scalar('Hits@10%', results['hits@10'], epoch)
            self.writer.add_scalar('MR'      , results['mr'], epoch)
            self.writer.add_scalar('MRR'     , results['mrr'], epoch)
        
        print(f"Epoch {epoch} validation:")
        for k, v in results.items():
            print(f"  {k}: {round(v, 5)}")

        return results['mrr']
    

    def _test_model(self, eval_method, bs):
        """
        Evaluate model on the test set

        Parameters:
        -----------
            eval_method: str
                filtered or not
            bs: int
                batch size
        
        Returns:
        --------
        None
        """
        model_eval = Evaluation("test", self.data, self.data.inverse, eval_method=eval_method, bs=bs, device=self.device)
        test_results = model_eval.evaluate(self.model)
        
        print("\nTest Results:", flush=True)
        for k, v in test_results.items():
            print(f"  {k}: {round(v, 5)}")



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
                        num_negative=num_negative,
                        inverse=self.data.inverse
                    )
        elif train_method == "1-N":
            sampler = sampling.One_to_N(
                        self.data['train'], 
                        bs, 
                        self.data.num_entities, 
                        self.device,
                        inverse=self.data.inverse
                    )
        else:
            raise ValueError(f"Invalid train method `{train_method}`")
        
        return sampler



