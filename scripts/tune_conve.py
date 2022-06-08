import os
import torch
import random 
import optuna
import argparse
import numpy as np
from optuna.trial import TrialState

from kgpy import sampling
from kgpy import models
from kgpy import datasets
from kgpy.training import Trainer


# Randomness!!!
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


parser = argparse.ArgumentParser(description='KG model and params to run')

parser.add_argument("--dataset", help="Dataset to run it on", default='fb15k_237')
parser.add_argument("--optimizer", help='Optimizer to use when training', type=str, default="Adam")
parser.add_argument("--epochs", help="Number of epochs to run", default=200, type=int)
parser.add_argument("--batch-size", help="Batch size to use for training", default=128, type=int)
parser.add_argument("--test-batch-size", help="Batch size to use for testing and validation", default=256, type=int)
parser.add_argument("--lr", help="Learning rate to use while training", default=1e-4, type=float)
parser.add_argument("--train-type", help="Type of training method to use", type=str, default="1-N")
parser.add_argument("--decay", help="Decay function for LR of form C^epoch", type=float, default=None)

parser.add_argument("--label-smooth", help="label smoothing", default=0.1, type=float)
parser.add_argument("--loss", help="Loss function to use.", default="bce")
parser.add_argument("--negative-samples", help="Number of negative samples to using 1-K training", default=50, type=int)

parser.add_argument("--device", help="Device to run on", type=str, default="cuda")
parser.add_argument("--validation", help="Test on validation set every n epochs", type=int, default=5)
parser.add_argument("--early-stop", help="Number of validation scores to wait for an increase before stopping", default=4, type=int)
parser.add_argument("--checkpoint-dir", default=os.path.join(os.path.expanduser("~"), "kgpy", "checkpoints"))

parser.add_argument("--filters", help="conv filters", type=int, default=32)
parser.add_argument("--emb-dim", help="Latent dimension of embeddings", type=int, default=200)

parser.add_argument("--num-trials", help="Number of trial to run", default=100, type=int)
parser.add_argument("--warmup", help="Min number of steps to run before considering pruning", default=20, type=int)

args = parser.parse_args()

DEVICE  = args.device
DATASET = args.dataset.upper()



class Objective(object):
    def __init__(self, data, sampler):
        # Hold this implementation specific arguments as the fields of the class.
        self.data = data 
        self.sampler = sampler


    def __call__(self, trial):
        """
        Objective Function for hyperparameter optimizer

        Returns:
        --------
        Validation MRR for trial 
        """
        lr = trial.suggest_categorical("lr", [5e-4, 1e-3, 5e-3])
        lbl_smooth = trial.suggest_categorical("label_smooth", [0, 0.1])
        inp_drop = trial.suggest_float("input_drop", 0, 0.5)
        hid_drop = trial.suggest_float("hidden_drop", 0, 0.5)
        feat_drop = trial.suggest_float("feat_drop", 0, 0.5)
        decay = trial.suggest_categorical("decay", [0.99, 0.995, 1])

        model_args = {
            "emb_dim": args.emb_dim,
            "input_drop": inp_drop,
            "feat_drop": feat_drop,
            "hidden_drop": hid_drop,
            "filters": args.filters,
            "device": DEVICE
        }

        model = models.ConvE(self.data.num_entities, self.data.num_relations, **model_args)
        model = model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_keywords = {
            "validate_every": args.validation, 
            "non_train_batch_size": args.test_batch_size, 
            "early_stopping": args.early_stop, 
            "negative_samples": args.negative_samples,
            "label_smooth": lbl_smooth,
            "decay": decay,
            "sampler": self.sampler,

            # Tuning!!!!
            "test_model": False,
            "optuna_trial": trial,
        }

        model_trainer = Trainer(model, optimizer, self.data, args.checkpoint_dir)
        mrr = model_trainer.fit(args.epochs, args.batch_size, args.train_type, **train_keywords)

        return mrr


def main():
    data = getattr(datasets, DATASET)(inverse=True)

    # Allows us to only create it once
    sampler = sampling.One_to_N(
                data['train'], 
                args.batch_size, 
                data.num_entities,
                data.num_relations, 
                args.device,
                inverse=True,
            )

    study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=args.warmup),
            )

    study.optimize(Objective(data, sampler), n_trials=args.num_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
