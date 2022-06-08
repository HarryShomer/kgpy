import os
import torch
import random 
import argparse
import numpy as np

from kgpy import models
from kgpy import datasets
from kgpy.training import Trainer


# Randomness!!!
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


parser = argparse.ArgumentParser(description='KG model and params to run')

parser.add_argument("--dataset", help="Dataset to run it on", default='fb15k_237')
parser.add_argument("--optimizer", help='Optimizer to use when training', type=str, default="Adam")
parser.add_argument("--epochs", help="Number of epochs to run", default=400, type=int)
parser.add_argument("--batch-size", help="Batch size to use for training", default=128, type=int)
parser.add_argument("--test-batch-size", help="Batch size to use for testing and validation", default=256, type=int)
parser.add_argument("--lr", help="Learning rate to use while training", default=1e-4, type=float)
parser.add_argument("--train-type", help="Type of training method to use", type=str, default="1-N")
parser.add_argument("--inverse", help="Include inverse edges", action='store_true', default=False)
# parser.add_argument("--decay", help="Decay function for LR of form C^epoch", type=float, default=None)

parser.add_argument("--label-smooth", help="label smoothing", default=0.1, type=float)
parser.add_argument("--lp", help="LP regularization penalty to add to loss", type=int, default=None)
parser.add_argument("--lp-weights", help="LP regularization weights. Can give one or two.", nargs='+', default=None)
parser.add_argument("--dim", help="Latent dimension of entities and relations", type=int, default=None)
parser.add_argument("--loss", help="Loss function to use.", default="bce")
parser.add_argument("--negative-samples", help="Number of negative samples to using 1-K training", default=1, type=int)

parser.add_argument("--device", help="Device to run on", type=str, default="cuda")
parser.add_argument("--parallel", help="Whether to train on multiple GPUs in parallel", action='store_true', default=False)
parser.add_argument("--validation", help="Test on validation set every n epochs", type=int, default=3)
parser.add_argument("--early-stop", help="Number of validation scores to wait for an increase before stopping", default=15, type=int)
parser.add_argument("--checkpoint-dir", default=os.path.join(os.path.expanduser("~"), "kgpy", "checkpoints"))
parser.add_argument("--tensorboard", help="Whether to log to tensorboard", action='store_true', default=False)
parser.add_argument("--save-as", help="Model to save model as", default=None, type=str)
parser.add_argument("--save-every", help="Save model every n epochs", default=50, type=int)

parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--comp-func", type=str, default="corr")
parser.add_argument("--num-filters", type=int, default=200)
parser.add_argument("--layer1-drop", type=float, default=.3)
parser.add_argument("--layer2-drop", type=float, default=.3)
parser.add_argument("--gcn-drop", type=float, default=.1)
parser.add_argument("--conve-drop1", type=float, default=.3)
parser.add_argument("--conve-drop2", type=float, default=.3)

parser.add_argument("--decoder", default='conve')


args = parser.parse_args()

DATASET = args.dataset.upper()
DEVICE  = args.device

data = getattr(datasets, DATASET)(inverse=args.inverse)

model_args = {
    "device": DEVICE,
    "decoder": args.decoder.lower(),
    "layer1_drop": args.layer1_drop,
    "layer2_drop": args.layer2_drop,
    "gcn_drop": args.gcn_drop,
    "conve_drop1": args.conve_drop1,
    "conve_drop2": args.conve_drop2,
    "num_filters": args.num_filters,
    "comp_func": args.comp_func,
    "num_layers": args.num_layers,
}


edge_index, edge_type = data.get_edge_tensors(device=DEVICE)
model = models.CompGCN(data.num_entities, data.num_relations, edge_index, edge_type, **model_args)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_keywords = {
    "validate_every": args.validation, 
    "non_train_batch_size": args.test_batch_size, 
    "early_stopping": args.early_stop, 
    "negative_samples": args.negative_samples,
    "label_smooth": args.label_smooth,
    "save_every": args.save_every,
}

model_trainer = Trainer(model, optimizer, data, args.checkpoint_dir, tensorboard=args.tensorboard, model_name=args.save_as)
model_trainer.fit(args.epochs, args.batch_size, args.train_type, **train_keywords)
