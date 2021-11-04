import os
import torch
import argparse

import kgpy
from kgpy import models


parser = argparse.ArgumentParser(description='KG model and params to run')
parser.add_argument("--dataset", type=str, default="FB15K_237")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--run", help="Name of saved model file", type=str, default=None)
args = parser.parse_args()

DECODER = 'distmult'
DEVICE = args.device
checkpoint_dir = "/mnt/home/shomerha/kgpy/checkpoints"


data = getattr(kgpy.datasets, args.dataset.upper())(inverse=True)
edge_index, edge_type = data.get_edge_tensors(rand_edge_perc=0, device=DEVICE)

model = models.CompGCN(data.num_entities, data.num_relations, edge_index, edge_type, decoder=DECODER, device=DEVICE) #, num_filters=250)
model = model.to(DEVICE)

checkpoint = torch.load(os.path.join(checkpoint_dir, data.dataset_name, f"{args.run}.tar"), map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

print("Frobenius Norm:\n--------")


in_norm = (torch.norm(model.conv1.w_in.data) + torch.norm(model.conv2.w_in.data)) / 2
out_norm = (torch.norm(model.conv1.w_out.data) + torch.norm(model.conv2.w_out.data)) / 2
loop_norm = (torch.norm(model.conv1.w_loop.data) + torch.norm(model.conv2.w_loop.data)) / 2
print(f"In:   {in_norm:.2f}")
print(f"Out:  {out_norm:.2f}")
print(f"Loop: {loop_norm:.2f}")


# model = models.RGCN(data.num_entities, data.num_relations, edge_index, edge_type, rgcn_num_bases=100, device=args.device)
# model = model.to(DEVICE)

# checkpoint = torch.load(os.path.join(checkpoint_dir, data.dataset_name, f"{args.run}.tar"),  map_location=DEVICE)
# model.load_state_dict(checkpoint['model_state_dict'])

# print("Frobenius Norm:\n-----")
# print(f"In: {torch.norm(model.conv1.w_in.data)}")
# print(f"Out: {torch.norm(model.conv1.w_out.data)}")
# print(f"Loop: {torch.norm(model.conv1.w_loop.data)}")

