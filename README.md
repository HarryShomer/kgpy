## kgpy

Implementations of various knowledge graph embedding techniques.

Models implemented so far:

1. TransE
2. DistMult
3. ComplEx
4. RotatE
5. ConvE
6. CompGCN
7. RGCN
8. TuckER

Possible datasets to run on:

1. WN18 and WN18RR
2. FB15K and FB15K-237 
3. YAGO3-10


### Usage

Below is a minimal example showing how to train CompGCN on FB15K-237.

```
import kgpy
import torch

lr = 1e-3
epochs = 400
batch_size = 128
device = 'cuda'

# Get data. We are also including inverse/reciprocal triples
data = kgpy.datasets.FB15K_237(inverse=True)

# Create our model and move to the gpu
edge_index, edge_type = data.get_edge_tensors()
model = kgpy.models.CompGCN(data.num_entities, data.num_relations, edge_index, edge_type, decoder="conve", device=device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train using 1-N strategy
# Will also evaluate on validation and test sets
model_trainer = kgpy.Trainer(model, optimizer, data)
model_trainer.fit(epochs, batch_size, "1-N")
```

#### CLI
For quick usage you can run `kgpy/main.py` using a list of command line arguments.

In order to work you must run it as a module.
```
python -m kgpy.main [Insert CLI args]
```

The full list of CLI args can be found by running `python -m kgpy.main --help`.

NOTE: Running with the CLI is limited as not all options are available for each model.


### TODO:

1. Implement other KGEs
2. Better documentation
3. Provide cleaner implementation of BaseGNNModel


### Data

The data found in the `datasets` directory is via https://github.com/ZhenfengLei/KGDatasets.