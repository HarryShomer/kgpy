import os
import torch
import random
import numpy as np
from collections import defaultdict
from torch_geometric.utils import to_dense_adj

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "datasets")



class TestDataset(torch.utils.data.Dataset):
    """
    Dataset object for test data
    """
    def __init__(self, triplets, all_triplets, num_entities, inverse=True, only_sample=False, device='cpu'):
        self.device = device
        self.inverse = inverse
        self.triplets = triplets
        self.num_entities = num_entities
        self.only_sample = only_sample

        self._build_index(all_triplets)


    def __len__(self):
        """
        Length of dataset
        """
        return len(self.triplets)
    

    def _build_index(self, triplets):
        """
        Mapping of triplets for testing
        """
        self.index = defaultdict(list)

        for t in triplets:
            if self.inverse:
                self.index[(t[1], t[0])].append(t[2])
            else:
                self.index[("head", t[1], t[2])].append(t[0])
                self.index[("tail", t[1], t[0])].append(t[2])

        # Remove duplicates
        for k, v in self.index.items():
            self.index[k] = list(set(v))


    def __getitem__(self, index):
        """
        For inverse just returns info for the tail/subject
        For non-inverse we return for both the head and tail

        Parameters:
        -----------
            index: int
                index for specific triplet

        Returns:
        -------
        tuple
            - Tensor containing subject and relation 
            - object ix
            - Tensor versus all possible objects - whether a true fact
        """        
        triple = torch.LongTensor(self.triplets[index])
        rel_sub = torch.LongTensor([triple[1].item(), triple[0].item()])
        rel_obj = torch.LongTensor([triple[1].item(), triple[2].item()])

        # Labels for all possible objects for triplet (s, r, ?)
        if self.inverse:
            # NOTE: self.only_sample is a hack to only assign the current triple as the true label!
            # This should not be used when training! 
            # This was only added to aid me in some analysis
            possible_obj = [triple[2]] if self.only_sample else np.int32(self.index[(triple[1].item(), triple[0].item())])

            obj_label  = self.get_label(possible_obj)

            return rel_sub, triple[2], obj_label, triple
        
        # For both (s, r, ?) and (?, r, o)
        possible_obj  = np.int32(self.index[("tail", triple[1].item(), triple[0].item())])
        possible_sub  = np.int32(self.index[("head", triple[1].item(), triple[2].item())])
        obj_label  = self.get_label(possible_obj)
        sub_label  = self.get_label(possible_sub)

        return rel_sub, triple[2], obj_label, rel_obj, triple[0], sub_label, triple 


        
    def get_label(self, possible_obj):
        y = np.zeros([self.num_entities], dtype=np.float32)
        
        for o in possible_obj: 
            y[o] = 1.0
        
        return torch.FloatTensor(y)




class AllDataSet():
    """
    Base class for all possible datasets
    """
    def __init__(
        self, 
        dataset_name, 
        inverse=False, 
        relation_pos="middle",
        perc_rels=1,
        perc_ents=1
    ):
        self.dataset_name = dataset_name
        self.relation_pos = relation_pos.lower()
        self.inverse  = inverse
        self.perc_rels, self.perc_ents = perc_rels, perc_ents

        self.entity2idx, self.relation2idx = self._load_mapping()
        self.entities, self.relations = list(set(self.entity2idx)), list(set(self.relation2idx))

        # Updates self.entities and self.relations
        self.prune_dataset()

        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)

        self.triplets = {
            "train": self._load_triplets("train"),
            "valid": self._load_triplets("valid"),
            "test":  self._load_triplets("test")
        }

        if self.inverse:
            self.num_relations *= 2
        

    @property
    def all_triplets(self):
        return list(set(self.triplets['train'] + self.triplets['valid'] + self.triplets['test']))

    @property
    def num_non_inv_rels(self):
        if self.inverse:
            return int(self.num_relations / 2)
        else:
            return self.num_relations

    @property
    def adjacency(self):
        """
        Construct the adjacency matrix. 1 where (u, v) in G else 0
        
        Notes:
            - This does not include relation info!
            - Stores on cpu
        """
        edge_index, _ = self.get_edge_tensors()

        # Construct adjacency where 1 = Neighbor otherwise 0
        adj = to_dense_adj(edge_index).squeeze(0)
        adj = torch.where(adj > 0, 1, 0)

        return adj


    def __getitem__(self, key):
        """
        Get specific dataset split
        """
        if key == 'train':
            return self.triplets['train']
        if key == 'valid':
            return self.triplets['valid']
        if key == "test":
            return self.triplets['test']
        
        raise ValueError("No key with name", key)


    def _load_mapping(self):
        """
        Load the mappings for the relations and entities from file

        key = name
        value = id

        Returns:
        --------
        tuple
            dictionaries mapping an entity or relation to it's ID
        """
        entity2idx, relation2idx = {}, {}

        with open(os.path.join(DATA_DIR, self.dataset_name, "entity2id.txt"), "r") as f:
            for line in f:
                line_components = [l.strip() for l in line.split()]
                entity2idx[line_components[0]] = int(line_components[1])

        with open(os.path.join(DATA_DIR, self.dataset_name, "relation2id.txt"), "r") as f:
            for line in f:
                line_components = [l.strip() for l in line.split()]
                relation2idx[line_components[0]] = int(line_components[1])

        return entity2idx, relation2idx



    def _load_triplets(self, data_split):
        """
        Load the triplets for a given dataset and data split.

        Use mapping IDs to represent triplet components

        Parameters:
        -----------
            data_split: str 
                Which split of the data to load (train/test/validation)

        Returns:
        --------
        list
            contain tuples representing triplets
        """
        triplets = []

        # Make lookup O(1)
        ent_set = set(range(0, len(self.entities)))
        rel_set = set(range(0, len(self.relations)))

        with open(os.path.join(DATA_DIR, self.dataset_name, f"{data_split}.txt"), "r") as file:
            for line in file:
                fields = [l.strip() for l in line.split()]

                # Stored in file as "s, o, r" instead of "s, r, o"
                if self.relation_pos.lower() == "end":
                    fields[1], fields[2] = fields[2], fields[1]
                
                s = self.entity2idx[fields[0]]
                r = self.relation2idx[fields[1]]
                o = self.entity2idx[fields[2]]

                # This only matters when we are pruning the dataset
                # In that case we are only using a portion of either the relations or entities
                if s in ent_set and o in ent_set and r in rel_set:
                    triplets.append((s, r, o))
                    
                    if self.inverse:
                        triplets.append((o, r + self.num_relations, s))


        return triplets


    def get_edge_tensors(self, rand_edge_perc=0, device='cpu'):
        """
        Create the edge_index and edge_type from the training data 

        Create random edges by (if specified):
            - generate non inv edge
            - Create inv to go along with it (if needed) 

        Parameters:
        ----------
            rand_edge_perc: float
                Percentage of random edges to add. E.g. when .5 add .5m new edges (where m = # of edges)
            device: str
                device to put edge tensors on

        Returns:
        --------
        tuple of torch.Tensor
            edge_index, edge_type    
        """
        new_edges = 0
        edge_index, edge_type = [], []

        if self.inverse:
            non_inv_edges = [e for e in self.triplets['train'] if e[1] < self.num_non_inv_rels]
        else:
            non_inv_edges = self.triplets['train']

        num_rand_edges = int(len(non_inv_edges) * rand_edge_perc)

        for sub, rel, obj in self.triplets['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)
      
        # Add 'num_keep_edges' random edges
        while new_edges < num_rand_edges and num_rand_edges != 0:
            self._generate_rand_edge(edge_index, edge_type)
            new_edges += 1  

        edge_index	= torch.LongTensor(edge_index).to(device)
        edge_type = torch.LongTensor(edge_type).to(device)

        return edge_index.transpose(0, 1), edge_type



    def _generate_rand_edge(self, edge_index, edge_type):
        """
        Generate a single random edge. Modifies params in place!

        Parameters:
        -----------
            edge_index: torch.Tensor
                2xN tensor holding head and tail nodes
            edge_type: torch.Tensor
                1xN tensor holding relation type

        Returns:
        --------
        None
        """
        num_rels = int(self.num_relations / 2) if self.inverse else self.num_relations

        r = np.random.randint(num_rels)
        s = np.random.randint(self.num_entities)
        o = np.random.randint(self.num_entities)

        edge_index.append((s, o))
        edge_type.append(r)

        if self.inverse:
            edge_index.append((o, s))
            edge_type.append(r + num_rels)
        

    def prune_dataset(self):
        """
        Prune either the entities or relations in the dataset.

        Modifies instance param in-place!
            - self.entities, self.relations
            - self.num_entities, self.num_relations
        
        Returns:
        --------
        None
        """
        if self.perc_ents != 1 and self.perc_rels != 1:
            raise ValueError("Both perc_ents and perc_rels can't both < 1. Only one.")

        if self.perc_ents != 1:
            num_ents = len(self.entities)
            ents_to_sample = int(num_ents * self.perc_ents)

            self.entities = random.sample(range(0, num_ents), ents_to_sample)
            self.num_entities = len(self.entities)

        if self.perc_rels != 1:
            num_rels = len(self.relations)
            rels_to_sample = int(num_rels * self.perc_rels)

            self.relations = random.sample(range(0, num_rels), rels_to_sample)
            self.num_relations = len(self.relations)


    def neighbor_rels_for_entity(self):
        """
        Unique relations for entity 

        For heads, we add non-inverse
        For tails, we add inverse 
        """
        r_adj = {e: set() for e in range(self.num_entities)}

        for t in self.triplets['train']:
            if t[1] < self.num_non_inv_rels:
                r_adj[t[0]].add(t[1])
                r_adj[t[2]].add(t[1] + self.num_non_inv_rels)

        return r_adj

    
    def neighbor_ents_for_entity(self):
        """
        Neighboring entities for entity...those connected by some relation
        """
        e_adj = {e: set() for e in range(self.num_entities)}

        for t in self.triplets['train']:
            e_adj[t[0]].add(t[2])
            e_adj[t[2]].add(t[0])

        return e_adj  


    def neighbor_ent_rels_for_entity(self):
        """
        Neighboring (e, r) pairs for a given entity
        
        """
        er_adj = {e: set() for e in range(self.num_entities)}

        for t in self.triplets['train']:
            if t[1] < self.num_non_inv_rels:
                er_adj[t[0]].add((t[1], t[2]))
                er_adj[t[2]].add((t[1] + self.num_non_inv_rels, t[0]))

        return er_adj


#######################################################
#######################################################
#######################################################


class FB15K_237(AllDataSet):
    """
    Load the FB15k-237 dataset
    """
    def __init__(self, **kwargs):
        super().__init__("FB15K-237", **kwargs)


class WN18RR(AllDataSet):
    """
    Load the WN18RR dataset
    """
    def __init__(self, **kwargs):
        super().__init__("WN18RR", **kwargs)



class FB15K(AllDataSet):
    """
    Load the FB15k dataset
    """
    def __init__(self, **kwargs):
        super().__init__("FB15K", relation_pos="end", **kwargs)


class WN18(AllDataSet):
    """
    Load the WN18 dataset
    """
    def __init__(self, **kwargs):
        super().__init__("WN18", relation_pos="end", **kwargs)


class YAGO3_10(AllDataSet):
    """
    Load the YAGO3-10 dataset
    """
    def __init__(self, **kwargs):
        super().__init__("YAGO3-10", **kwargs)
