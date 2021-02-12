import os
import copy
import random
import torch
import numpy as np

import utils
from load_data import TrainDataset, TestDataset
from evaluation import evaluate_model


if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


class Trainer:

    def __init__(self, model, optimizer, data, tensorboard_writer, log_every_n_steps=25):
        self.model = model
        self.optimizer = optimizer
        self.data = data 
        self.writer = tensorboard_writer
        self.log_every_n_steps = log_every_n_steps


    def train(self, epochs, train_batch_size, validate_every=5, val_batch_size=1, early_stopping=5, save_every=25, negative_samples=2):
        """
        Train and validate the model

        Args:
            epochs: 
                Number of epochs to train for
            train_batch_size: 
                Batch size to use for training
            validate_every: 
                Validate every "n" epochs. Defaults to 5
            val_batch_size: 
                Batch size for non-training data. Defaults to 16
            early_stopping: 
                Stop training if the mean rank hasn't improved in last "n" validation scores. Defaults to 5
            save_every: 
                Save model every "n" epochs. Defaults to 50

        Returns:
            None
        """
        step = 1
        val_mean_ranks = []
        train_loader = torch.utils.data.DataLoader(
                           TrainDataset(self.data.dataset_name, self.data.train_triplets), 
                           batch_size=train_batch_size
                       )

        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}")
            self.model.train()   # Switch back to train from eval

            for batch in train_loader:
                batch_heads, batch_relations, batch_tails = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                triplets = torch.stack((batch_heads, batch_relations, batch_tails), dim=1)
                corrupted_triplets = self.corrupt_triplets(triplets)

                # corrupted_triplets = self.corrupt_triplets(triplets, negative_samples)
                # triplets =  triplets.repeat(negative_samples, 1)

                self.optimizer.zero_grad()

                batch_loss = self.model(triplets.detach(), corrupted_triplets.detach())
                batch_loss.backward()

                self.optimizer.step()
                step += 1

                if step % self.log_every_n_steps == 0:
                    self.writer.add_scalar(f'training_loss', batch_loss.item(), global_step=step)


            if epoch % validate_every == 0:
                val_mean_ranks.append(self.validate_model(epoch, val_batch_size))
    
                # Start checking after accumulate more than val mean rank
                if len(val_mean_ranks) >= early_stopping and np.argmin(val_mean_ranks[-early_stopping:]) == 0:
                    print(f"Validation loss hasn't improved in the last {early_stopping} validation mean rank scores. Stopping training now!")
                    break

                # Only save when we know the model performs better
                utils.save_model(self.model, self.optimizer, epoch, step, self.data.dataset_name)


            if epoch % save_every == 0:
                utils.save_model(self.model, self.optimizer, epoch, step, self.data.dataset_name, suffix=f"epoch_{epoch}")



    def validate_model(self, epoch, batch_size):
        """
        Test model on validation and log to tensorboard

        Args:
            epoch:
                Current training epoch
            batch_size:
                Batch size used for testing

        Return:
            mean rank on validation set
        """
        dataloader = torch.utils.data.DataLoader(
                        TestDataset(self.data.dataset_name, self.data.valid_triplets, self.data.all_triplets, self.data.num_entities), 
                        batch_size=batch_size,
                        num_workers=8
                    )
        mr, mrr, hits_at_1, hits_at_3, hits_at_10 = evaluate_model(self.model, dataloader)

        # Only save when we know the model performs better
        self.writer.add_scalar('Hits@1%' , hits_at_1, epoch)
        self.writer.add_scalar('Hits@3%' , hits_at_3, epoch)
        self.writer.add_scalar('Hits@10%', hits_at_10, epoch)
        self.writer.add_scalar('MR'      , mr, epoch)
        self.writer.add_scalar('MRR'     , mrr, epoch)

        return mr


    # def corrupt_triplets(self, triplets, negative_samples):
    #     """
    #     Corrupt list of triplet by randomly replacing either the head or the tail with another entitiy

    #     Args:
    #         triplets: list of triplet to corrupt 

    #     Returns:
    #         Corrupted Triplets
    #     """
    #     corrupted_triplets = []


    #     for _ in range(negative_samples):

    #         for i, t in enumerate(triplets):
            
    #             new_triplet = copy.deepcopy(t)
    #             head_tail = random.choice([0, 2])
    #             new_triplet[head_tail] = utils.randint_exclude(0, len(self.data.entities), t[head_tail])
                
    #             corrupted_triplets.append(new_triplet)


    #     return torch.stack(corrupted_triplets, dim=0)



    def corrupt_triplets(self, triplets):
        """
        Corrupt list of triplet by randomly replacing either the head or the tail with another entitiy
        Args:
            triplets: list of triplet to corrupt 
        Returns:
            Corrupted Triplets
        """
        corrupted_triplets = copy.deepcopy(triplets)

        for i, t in enumerate(triplets):
            head_tail = random.choice([0, 2])
            corrupted_triplets[i][head_tail] = utils.randint_exclude(0, len(self.data.entities), t[head_tail])

        return corrupted_triplets

