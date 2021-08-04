import os
import torch
import numpy as np
from collections import defaultdict


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "kg_datasets")




class TestDataset(torch.utils.data.Dataset):
    """
    Dataset object for test data
    """

    def __init__(self, triplets, all_triplets, num_entities, inverse=True, device='cpu'):
        self.device = device
        self.inverse = inverse
        self.triplets = triplets
        self.num_entities = num_entities

        # self.all_triplets = {t: True for t in all_triplets}

        if not inverse:
            raise NotImplementedError("TODO: Not implemented non-inverse test dataloader yet")
        
        self._build_index(all_triplets)


    def __len__(self):
        """
        Length of dataset
        """
        return len(self.triplets)


    def _build_index(self, triplets):
        """
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
        Override prev implementation to help with filtered metrics

        Add dimension to triplet to indicate whether it is a true triplet or not.
        
        True is denoted by 0 and False by 1
        """
        # corrupted_head_triplets = []
        # corrupted_tail_triplets = []
        # head, relation, tail = self.triplets[index]

        # for e in range(self.num_entities):
        #     corrupt_head = (e, relation, tail)
        #     corrupt_tail = (head, relation, e)

        #     if self.evaluation_method == "filtered":
        #         corrupt_head_bit = int(self.all_triplets.get(corrupt_head) is None)
        #         corrupt_tail_bit = int(self.all_triplets.get(corrupt_tail) is None)
        #     else:
        #         corrupt_head_bit = (e, relation, tail) != self.triplets[index]
        #         corrupt_tail_bit = (head, relation, e) != self.triplets[index]

        #     corrupted_head_triplets.append(corrupt_head + (corrupt_head_bit,))
        #     corrupted_tail_triplets.append(corrupt_tail + (corrupt_tail_bit,))


        # return torch.LongTensor(self.triplets[index]), torch.LongTensor(corrupted_head_triplets), torch.LongTensor(corrupted_tail_triplets)

        # TODO: Implement for head/tail
        # Only works for inverse here

        triple = torch.LongTensor(self.triplets[index])
        
        rel_sub = torch.LongTensor([triple[1].item(), triple[0].item()])
        possible_obj  = np.int32(self.index[(triple[1].item(), triple[0].item())])

        label  = self.get_label(possible_obj)

        return rel_sub, triple[2], label


    @staticmethod
    def collate_fn(self, data):
        triple	= torch.stack([_[0] for _ in data], dim=0)
        obj		= torch.stack([_[1] for _ in data], dim=0)
        label	= torch.stack([_[2] for _ in data], dim=0)

        return triple, obj, label


    def get_label(self, possible_obj):
        y = np.zeros([self.num_entities], dtype=np.float32)
        
        for o in possible_obj: 
            y[o] = 1.0
        
        return torch.FloatTensor(y)




class AllDataSet():
    """
    Base class for all possible datasets
    """

    def __init__(self, dataset_name, inverse=False, relation_pos="middle"):
        self.dataset_name = dataset_name
        self.relation_pos = relation_pos
        self.inverse  = inverse

        self.entity2idx, self.relation2idx = self._load_mapping()
        self.entities, self.relations = list(set(self.entity2idx)), list(set(self.relation2idx))

        self.num_relations = len(self.relations)

        self.triplets = {
            "train": self._load_triplets("train"),
            "valid": self._load_triplets("valid"),
            "test":  self._load_triplets("test")
        }

        if self.inverse:
            self.num_relations *= 2



    @property
    def num_entities(self):
        return len(self.entities)
    

    @property
    def all_triplets(self):
        return list(set(self.triplets['train'] + self.triplets['valid'] + self.triplets['test']))

        
    def __getitem__(self, key):
        """
        Get specific dataset
        """
        if key == 'train':
            return self.triplets['train']
        if key == 'valid':
            return self.triplets['valid']
        if key == "test":
            return self.triplets['test']
        print("No key with name", key)

        return None


    def all_triplets_map(self):
        """
        Not a property for efficiency
        """
        return {t: True for t in self.all_triplets}


    def _load_mapping(self):
        """
        Load the mappings for the relations and entities from file

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

        with open(os.path.join(DATA_DIR, self.dataset_name, f"{data_split}.txt"), "r") as file:
            for line in file:
                fields = [l.strip() for l in line.split()]

                # When relation not in middle swap it there
                if self.relation_pos.lower() != "middle":
                    fields[1], fields[2] = fields[2], fields[1]

                triplets.append((self.entity2idx[fields[0]], self.relation2idx[fields[1]], self.entity2idx[fields[2]]))
            
                if self.inverse:
                    triplets.append((self.entity2idx[fields[0]], self.relation2idx[fields[1]] + self.num_relations, self.entity2idx[fields[2]]))


        return triplets





#######################################################
#######################################################
#######################################################


class FB15K_237(AllDataSet):
    """
    Load the FB15k-237 dataset
    """
    def __init__(self, inverse=False):
        super().__init__("FB15K-237", inverse=inverse)


class WN18RR(AllDataSet):
    """
    Load the WN18RR dataset
    """
    def __init__(self, inverse=False):
        super().__init__("WN18RR", inverse=inverse)



class FB15K(AllDataSet):
    """
    Load the FB15k dataset
    """
    def __init__(self, inverse=False):
        super().__init__("FB15K", relation_pos="end", inverse=inverse)


class WN18(AllDataSet):
    """
    Load the WN18 dataset
    """
    def __init__(self, inverse=False):
        super().__init__("WN18", relation_pos="end", inverse=inverse)