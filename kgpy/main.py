import os
import torch
import argparse
import numpy as np 
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

import utils
import models
import load_data
import evaluation
from training import Trainer


if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


#parser = argparse.ArgumentParser(description='')
#parser.add_argument('-t', "--reportType", help='Type of report to scrape. Either game or schedule.', default='game', type=str, required=False)  
#parser.add_argument("--shifts", help='Whether to include shifts.', action='store_true', default=False, required=False)


# Constants
EPOCHS = 500
TRAIN_BATCH_SIZE = 128
TEST_VAL_BATCH_SIZE = 16
LEARNING_RATE = 0.0001
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
    
    # print(f"\nTest Results - Last Saved:")
    # evaluation.test_model(model, data, batch_size=TEST_VAL_BATCH_SIZE)

    # Now let's see how they did by epoch
    for i in range(25, 1050, 50):
        if not utils.checkpoint_exists(model.name, data.dataset_name, epoch=i):
            print(f"The model checkpoint for {model.name} at epoch {i} was never saved.")
            continue
   
        model, optimizer = utils.load_model(model, optimizer, data.dataset_name, epoch=i)

        print(f"\nTest Results - Epoch {i}:")
        evaluation.test_model(model, data, batch_size=TEST_VAL_BATCH_SIZE)




def run_model(model, optimizer, data):
    """
    Wrapper for training and testing the model

    Args:
        model: 
            pytorch model
        optimizer:
            pytorch optimizer
        data:
            AllDataset object

    Returns:
        None
    """
    tensorboard_writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, model.name, data.dataset_name), flush_secs=3)

    model_trainer = Trainer(model, optimizer, data, tensorboard_writer)
    model_trainer.train(EPOCHS, TRAIN_BATCH_SIZE)

    print("\nTest Results:")
    evaluation.test_model(model, data, TEST_VAL_BATCH_SIZE)


  

def main():
    #data = load_data.FB15k_237()
    data = load_data.WN18RR()

    #model = models.DistMult(data.entities, data.relations, l2=5e-6, latent_dim=156)
    model = models.TransE(data.entities, data.relations, latent_dim=256)

    model = model.to(device)

    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=model.l2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=model.l2)

    run_model(model, optimizer, data)
    #test_diff_models(model, optimizer, data)



if __name__ == "__main__":
    main()
