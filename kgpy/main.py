import os
import numpy as np 
import torch
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

import load_data
import create_model
import utils
from evaluation import test_model


if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"

# Constants
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = .01
EVERY_N_EPOCHS_VAL = 5    # Test on validation set every N epochs
EVERY_N_STEPS_TRAIN = 25  # Write training loss to tensorboard every N steps
LAST_N_VAL = 4            # Compare validation metric to last N scores. If it hasn't decreased in that time we stop training.
    
TENSORBOARD_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "runs")


def test_diff_models(data):
    """
    """
    test_loader = torch.utils.data.DataLoader(data.test, batch_size=16)

    for i in range(50, 1000, 50):
        model = create_model.TransE(data.entities, data.relations)
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

        model, optimizer = utils.load_model(model, optimizer, i, data.dataset_name)

        print(f"\nTest Results - Epoch {i}:")
        mr, mrr, hits_at_1, hits_at_3, hits_at_10 = test_model(model, test_loader, data.num_entities)
        print(f"MR: {mr} \nMRR: {mrr} \nhits@1: {hits_at_1} \nhits@3: {hits_at_3} \nhits@10: {hits_at_10}\n")




def train(model, data):
    """

    Args:
        model:
        data:

    Returns:

    """
    summary_writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, model.name, data.dataset_name))

    train_loader = torch.utils.data.DataLoader(data.train, batch_size=BATCH_SIZE)
    valid_loader = torch.utils.data.DataLoader(data.validation, batch_size=16)
    test_loader = torch.utils.data.DataLoader(data.test, batch_size=16)

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    step = 1
    val_mean_ranks = []

    for epoch in range(1, EPOCHS+1):
        print(f"Epoch {epoch}")
        model.train()   # Switch back to train from eval

        for batch in train_loader:
            batch_heads, batch_relations, batch_tails = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            triplets = torch.stack((batch_heads, batch_relations, batch_tails), dim=1)
            corrupted_triplets = model.corrupt_triplets(triplets)

            optimizer.zero_grad()

            batch_loss = model(triplets.detach(), corrupted_triplets.detach())
            batch_loss.backward()

            optimizer.step()
            step += 1

            if step % EVERY_N_STEPS_TRAIN == 0:
                summary_writer.add_scalar(f'training_loss', batch_loss.item(), global_step=step)


        # Save and test model on validation every every_n_epochs epochs
        # When validation hasn't improved in last `last_n_val` validation mean ranks
        if epoch % EVERY_N_EPOCHS_VAL == 0:
            mr, mrr, hits_at_1, hits_at_3, hits_at_10 = test_model(model, valid_loader, data.num_entities)

            # Only save when we know the model performs better
            summary_writer.add_scalar('Hits@1' , hits_at_1, epoch)
            summary_writer.add_scalar('Hits@3' , hits_at_3, epoch)
            summary_writer.add_scalar('Hits@10' , hits_at_10, epoch)
            summary_writer.add_scalar('MR'     , mr, epoch)
            summary_writer.add_scalar('MRR'    , mrr, epoch)

            val_mean_ranks.append(mr)
        
            # Early stopping
            # Start checking after accumulate more than val mean rank
            if len(val_mean_ranks) >= LAST_N_VAL and np.argmin(val_mean_ranks[-LAST_N_VAL:]) == 0:
                print(f"Validation loss hasn't improved in the last {LAST_N_VAL} validation mean rank scores. Stopping training now!")
                break

            # Only save when we know the model performs better
            utils.save_model(model, optimizer, epoch, step, data.dataset_name)

        # Save every 50 epochs because why not
        if epoch % 50 == 0:
            utils.save_model(model, optimizer, epoch, step, data.dataset_name, suffix=f"epoch_{epoch}")


    # Tet results
    print("\nTest Results:")
    mr, mrr, hits_at_1, hits_at_3, hits_at_10 = test_model(model, test_loader, data.num_entities)
    print(f"MR: {mr} \nMRR: {mrr} \nhits@1: {hits_at_1} \nhits@3: {hits_at_3} \nhits@10: {hits_at_10}\n")


   

def main():
    data = load_data.FB15k_237()
    transe = create_model.TransE(data.entities, data.relations)
    train(transe, data)
    #test_diff_models(wn_data)



if __name__ == "__main__":
    main()
