import os
import torch
import argparse
import numpy as np 
from torch.utils import tensorboard

import utils
import models
import load_data
import evaluation
from training import Trainer


if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


parser = argparse.ArgumentParser(description='KG model and params to run')
parser.add_argument("model", help="Model to run")
parser.add_argument("dataset", help="Dataset to run it on")
parser.add_argument("--epochs", help="Number of epochs to run", default=500, type=int)
parser.add_argument("--batch-size", help="Batch size to use for training", default=128, type=int)
parser.add_argument("--learning-rate", help="Learning rate to use while training", default=.0001, type=float)

parser.add_argument("--lp", help="LP regularization penalty to add to loss", type=int, default=None)
parser.add_argument("--lp-weights", help="LP regularization weights. Can give one or two.", nargs='+', default=None)
parser.add_argument("--dim", help="Latent dimension of entities and relations", type=int, default=None)
parser.add_argument("--loss-fn", help="Loss function to use.", default=None)

parser.add_argument("--test-batch-size", help="Batch size to use for testing and validation", default=16, type=int)
parser.add_argument("--validation", help="Test on validation set every n epochs", type=int, default=5)
parser.add_argument("--early-stopping", help="Number of validation scores to wait for an increase before stopping", default=5, type=int)
parser.add_argument("--checkpoint-dir", help="Directory to store model checkpoints", default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "checkpoints"))
parser.add_argument("--tensorboard", help="Whether to log to tensorboard", action='store_true', default=False)
parser.add_argument("--log-training-loss", help="Log training loss every n steps", default=25, type=int)

## TODO:
# margin

args = parser.parse_args()


# Constants
EPOCHS = args.epochs
TRAIN_BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
EVERY_N_EPOCHS_VAL = args.validation   
LAST_N_VAL = args.early_stopping            
TEST_VAL_BATCH_SIZE = args.test_batch_size    # Batch size for test and validation testing
EVERY_N_STEPS_TRAIN = args.log_training_loss  # Write training loss to tensorboard every N steps
CHECKPOINT_DIR = args.checkpoint_dir  



def test_diff_models(model, optimizer, data):
    """
    Test different versions of a given model (at different epochs).

    Also test the final main version (likely differs from last epoch version)
    """
    model, optimizer = utils.load_model(model, optimizer, data.dataset_name)
    
    print(f"\nTest Results - Last Saved:")
    evaluation.test_model(model, data, batch_size=TEST_VAL_BATCH_SIZE)

    # Now let's see how they did by epoch
    for i in range(25, 1050, 25):
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
    model_trainer = Trainer(model, optimizer, data, CHECKPOINT_DIR, tensorboard=args.tensorboard)
    model_trainer.train(EPOCHS, TRAIN_BATCH_SIZE)

    print("\nTest Results:")
    evaluation.test_model(model, data, TEST_VAL_BATCH_SIZE)


def parse_model_args():
    """
    Parse cmd line args to create the model.

    They are only added when passed (aka not None)

    Returns: dict
        Keyword arguments for model 
    """
    model_params = {}


    if args.lp is not None:
        model_params['regularization'] = f"l{args.lp}"

    if isinstance(args.lp_weights, list):
        model_params['reg_weight'] = [float(r) for r in args.lp_weights]
    elif args.lp_weights is not None:
        model_params['reg_weight'] = float(args.lp_weights)

    if args.dim is not None:
        model_params['latent_dim'] = args.dim

    if args.loss_fn is not None:
        model_params['loss_fn'] = args.loss_fn 
       
    return model_params
  

  

def main():
    data = getattr(load_data, args.dataset.upper())()

    model_name = args.model.lower()
    model_params = parse_model_args()

    # TODO: Convert to argparse
    if model_name == "transe":
        model = models.TransE(data.entities, data.relations, **model_params)
    if model_name == "distmult":
        model = models.DistMult(data.entities, data.relations, **model_params)
    if model_name == "complex":  
        model = models.ComplEx(data.entities, data.relations, **model_params)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    run_model(model, optimizer, data)

    #test_diff_models(model, optimizer, data)



if __name__ == "__main__":
    main()
