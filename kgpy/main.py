import os
import numpy as np 
import torch
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

import load_data
import models
import utils
from evaluation import test_model, validate_model


if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


# Constants
EPOCHS = 1000
TRAIN_BATCH_SIZE = 128
TEST_VAL_BATCH_SIZE = 16
LEARNING_RATE = 0.001
EVERY_N_EPOCHS_VAL = 5    # Test on validation set every N epochs
EVERY_N_STEPS_TRAIN = 25  # Write training loss to tensorboard every N steps
LAST_N_VAL = 5            # Compare validation metric to last N scores. If it hasn't decreased in that time we stop training.
    
TENSORBOARD_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "runs")


def test_diff_models(model, optimizer, data):
    """
    Test different versions of a given model (at different epochs).

    Also test the final main version (likely differs from last epoch version)
    """
    model, optimizer = utils.load_model(model, optimizer, data.dataset_name)
    
    print(f"\nTest Results - Last Saved:")
    test_model(model, data, batch_size=TEST_VAL_BATCH_SIZE)

    # Now let's see how they did by epoch
    for i in range(50, 1050, 50):
        if not utils.checkpoint_exists(model.name, data.dataset_name, epoch=i):
            print(f"The model checkpoint for {model.name} at epoch {i} was never saved.")
            continue
   
        model, optimizer = utils.load_model(model, optimizer, data.dataset_name, epoch=i)

        print(f"\nTest Results - Epoch {i}:")
        test_model(model, data, batch_size=TEST_VAL_BATCH_SIZE)



def train(model, optimizer, data):
    """
    Train and validate the model

    Args:
        model:
        optimizer
        data:

    Returns:
        None
    """
    summary_writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, model.name, data.dataset_name), flush_secs=3)

    train_loader = torch.utils.data.DataLoader(data.train, batch_size=TRAIN_BATCH_SIZE)

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
            mr, mrr, hits_at_1, hits_at_3, hits_at_10 = validate_model(model, data, batch_size=TEST_VAL_BATCH_SIZE)

            # Only save when we know the model performs better
            summary_writer.add_scalar('Hits@1%' , hits_at_1, epoch)
            summary_writer.add_scalar('Hits@3%' , hits_at_3, epoch)
            summary_writer.add_scalar('Hits@10%', hits_at_10, epoch)
            summary_writer.add_scalar('MR'      , mr, epoch)
            summary_writer.add_scalar('MRR'     , mrr, epoch)

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
    test_model(model, data, batch_size=TEST_VAL_BATCH_SIZE)


   

def main():
    data = load_data.FB15k_237()
    #data = load_data.WN18RR()

    #model = models.DistMult(data.entities, data.relations)
    model = models.TransE(data.entities, data.relations)

    model = model.to(device)

    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=model.l2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=model.l2)

    train(model, optimizer, data)
    #test_diff_models(model, optimizer, data)



if __name__ == "__main__":
    main()
