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


if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


TENSORBOARD_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "runs")


class Trainer:
    """
    Control training of model on a particular dataset
    """

    def __init__(self, model, optimizer, data, checkpoint_dir, tensorboard=True):
        self.model = model
        self.optimizer = optimizer
        self.data = data 
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard = tensorboard

        if tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, model.name, data.dataset_name), flush_secs=3)


    def train(
            self, 
            epochs, 
            train_batch_size, 
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
        val_mean_ranks = []
        train_loader = torch.utils.data.DataLoader(
                           TrainDataset(self.data.dataset_name, self.data.train_triplets), 
                           batch_size=train_batch_size
                       )

        for epoch in range(1, epochs+1):
            prog_bar = tqdm(train_loader, file=sys.stdout)
            prog_bar.set_description(f"Epoch {epoch}")

            self.model.train()   # Switch back to train from eval

            for batch in prog_bar:
                batch_heads, batch_relations, batch_tails = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                triplets = torch.stack((batch_heads, batch_relations, batch_tails), dim=1)
                
                corrupted_triplets = self.corrupt_triplets(triplets, negative_samples)
                triplets =  triplets.repeat(negative_samples, 1)

                self.optimizer.zero_grad()

                batch_loss = self.model(triplets, corrupted_triplets)
                batch_loss = batch_loss.mean()
                batch_loss.backward()

                self.optimizer.step()
                step += 1

                if step % log_every_n_steps == 0 and self.tensorboard:
                    self.writer.add_scalar(f'training_loss', batch_loss.item(), global_step=step)


            if epoch % validate_every == 0:
                val_mean_ranks.append(self.validate_model(epoch, non_train_batch_size, evaluation_method))
    
                # Start checking after accumulate more than val mean rank
                if len(val_mean_ranks) >= early_stopping and np.argmin(val_mean_ranks[-early_stopping:]) == 0:
                    print(f"Validation loss hasn't improved in the last {early_stopping} validation mean rank scores. Stopping training now!", flush=True)
                    break

                # Only save when we know the model performs better
                utils.save_model(self.model, self.optimizer, epoch, step, self.data.dataset_name, self.checkpoint_dir)


            if epoch % save_every == 0:
                utils.save_model(self.model, self.optimizer, epoch, step, self.data.dataset_name, self.checkpoint_dir, suffix=f"epoch_{epoch}")



    def validate_model(self, epoch, batch_size, evaluation_method):
        """
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

        return mr


    # def corrupt_triplets(self, triplets):
    #     """
    #     Corrupt list of triplet by randomly replacing either the head or the tail with another entitiy

    #     Args:
    #         triplets: list of triplet to corrupt 

    #     Returns:
    #         Corrupted Triplets
    #     """
    #     corrupted_triplets = copy.deepcopy(triplets)

    #     for i, t in enumerate(triplets):
    #         head_tail = random.choice([0, 2])
    #         corrupted_triplets[i][head_tail] = utils.randint_exclude(0, len(self.data.entities), t[head_tail])

    #     return corrupted_triplets



    def corrupt_triplets(self, triplets, negative_samples):
        """
        Corrupt list of triplet by randomly replacing either the head or the tail with another entitiy

        Parameters:
        -----------
            triplets: list 
                triplets to corrupt 

        Returns:
        --------
        list
            Corrupted Triplets
        """
        corrupted_triplets = []


        for _ in range(negative_samples):

            for i, t in enumerate(triplets):
            
                new_triplet = copy.deepcopy(t)
                head_tail = random.choice([0, 2])
                new_triplet[head_tail] = utils.randint_exclude(0, len(self.data.entities), t[head_tail])
                
                corrupted_triplets.append(new_triplet)


        return torch.stack(corrupted_triplets, dim=0)
