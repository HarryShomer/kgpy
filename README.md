## kgpy

Models implemented so far:

1. TransE
2. DistMult
3. ComplEx

Possible datasets to run on:

1. WN18 and WN18RR
2. FB15K and FB15K-237 


### Usage

You can run `kgpy/main.py` with the following command line argument options.

```
usage: main.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE] [--lp LP]
               [--lp-weights LP_WEIGHTS [LP_WEIGHTS ...]] [--dim DIM] [--loss-fn LOSS_FN] [--negative-samples NEGATIVE_SAMPLES]
               [--test-batch-size TEST_BATCH_SIZE] [--validation VALIDATION] [--early-stopping EARLY_STOPPING]
               [--checkpoint-dir CHECKPOINT_DIR] [--tensorboard] [--log-training-loss LOG_TRAINING_LOSS] [--test-model]
               model dataset

KG model and params to run

positional arguments:
  model                 Model to run
  dataset               Dataset to run it on

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs to run
  --batch-size BATCH_SIZE
                        Batch size to use for training
  --learning-rate LEARNING_RATE
                        Learning rate to use while training
  --lp LP               LP regularization penalty to add to loss
  --lp-weights LP_WEIGHTS [LP_WEIGHTS ...]
                        LP regularization weights. Can give one or two.
  --dim DIM             Latent dimension of entities and relations
  --loss-fn LOSS_FN     Loss function to use.
  --negative-samples NEGATIVE_SAMPLES
                        Number of negative samples to use when training
  --test-batch-size TEST_BATCH_SIZE
                        Batch size to use for testing and validation
  --validation VALIDATION
                        Test on validation set every n epochs
  --early-stopping EARLY_STOPPING
                        Number of validation scores to wait for an increase before stopping
  --checkpoint-dir CHECKPOINT_DIR
                        Directory to store model checkpoints
  --tensorboard         Whether to log to tensorboard
  --log-training-loss LOG_TRAINING_LOSS
                        Log training loss every n steps
  --test-model          Evaluate all saved versions of a given model and dataset on the test set
```

In order to work you must run it as a module. This is done like so
```
python -m kgpy.main [Insert CLI args]
```

### TODO:

1. Negative Sampling // Refactor?
2. Create custom classes for loss functions // loss.py
3. Implement other baseline methods (e.g. ConvE)
4. Better documentation

### Data

The data found in the `kg_datasets` directory is via https://github.com/ZhenfengLei/KGDatasets.