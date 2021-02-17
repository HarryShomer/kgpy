import os
import copy
import numpy as np
import torch
import random
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

from evaluation import evaluate_model
import utils


if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


TENSORBOARD_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "runs")


class Trainer:

    def __init__(self, model, optimizer, data, checkpoint_dir, tensorboard=True, log_every_n_steps=25):
        self.model = model
        self.optimizer = optimizer
        self.data = data 
        self.checkpoint_dir = checkpoint_dir
        self.log_every_n_steps = log_every_n_steps
        self.tensorboard = tensorboard

        if tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, model.name, data.dataset_name), flush_secs=3)


    def train(self, epochs, train_batch_size, validate_every=5, non_train_batch_size=16, early_stopping=5, save_every=25, negative_samples=5):
        """
        Train and validate the model

        Args:
            epochs: 
                Number of epochs to train for
            train_batch_size: 
                Batch size to use for training
            validate_every: 
                Validate every "n" epochs. Defaults to 5
            non_train_batch_size: 
                Batch size for non-training data. Defaults to 16
            early_stopping: 
                Stop training if the mean rank hasn't improved in last "n" validation scores. Defaults to 5
            save_Every: 
                Save model every "n" epochs. Defaults to 50
            negative_samples:
                Number of negative samples to generate for each training sample

        Returns:
            None
        """
        step = 1
        val_mean_ranks = []
        train_loader = torch.utils.data.DataLoader(self.data.train, batch_size=train_batch_size)


        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}")
            self.model.train()   # Switch back to train from eval

            for batch in train_loader:
                batch_heads, batch_relations, batch_tails = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                triplets = torch.stack((batch_heads, batch_relations, batch_tails), dim=1)
                
                #corrupted_triplets = self.corrupt_triplets(triplets)


                corrupted_triplets = self.corrupt_triplets(triplets, negative_samples)
                triplets =  triplets.repeat(negative_samples, 1)


                self.optimizer.zero_grad()

                batch_loss = self.model(triplets.detach(), corrupted_triplets.detach())
                batch_loss.backward()

                self.optimizer.step()
                step += 1

                if step % self.log_every_n_steps == 0 and self.tensorboard:
                    self.writer.add_scalar(f'training_loss', batch_loss.item(), global_step=step)


            if epoch % validate_every == 0:
                val_mean_ranks.append(self.validate_model(epoch, non_train_batch_size))
    
                # Start checking after accumulate more than val mean rank
                if len(val_mean_ranks) >= early_stopping and np.argmin(val_mean_ranks[-early_stopping:]) == 0:
                    print(f"Validation loss hasn't improved in the last {early_stopping} validation mean rank scores. Stopping training now!")
                    break

                # Only save when we know the model performs better
                utils.save_model(self.model, self.optimizer, epoch, step, self.data.dataset_name, self.checkpoint_dir)


            if epoch % save_every == 0:
                utils.save_model(self.model, self.optimizer, epoch, step, self.data.dataset_name, self.checkpoint_dir, suffix=f"epoch_{epoch}")



    def validate_model(self, epoch, non_train_batch_size):
        """
        """
        dataloader = torch.utils.data.DataLoader(self.data['validation'], batch_size=non_train_batch_size)
        mr, mrr, hits_at_1, hits_at_3, hits_at_10 = evaluate_model(self.model, dataloader, self.data)

        if self.tensorboard:
            # Only save when we know the model performs better
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

        Args:
            triplets: list of triplet to corrupt 

        Returns:
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
