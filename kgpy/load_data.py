import os
import json
import torch
import random
import numpy as np 

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "kg_datasets")

if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"



class TrainDataset(torch.utils.data.Dataset):
    """
    Extend torch.utils.data.Dataset.

    Loads a given split for a data set (e.g. training data)
    """

    def __init__(self, dataset_name, triplets):
        self.dataset_name = dataset_name
        self.triplets = triplets


    def __len__(self):
        """
        Length of dataset
        """
        return len(self.triplets)


    def __getitem__(self, index):
        """
        Get indicies for the ith triplet
        """
        return self.triplets[index][0], self.triplets[index][1], self.triplets[index][2]



class TestDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, triplets, all_triplets, num_entities):
        self.dataset_name = dataset_name

        self.triplets = triplets
        self.num_entities = num_entities

        self.all_triplets = {t: True for t in all_triplets}


    def __len__(self):
        """
        Length of dataset
        """
        return len(self.triplets)


    def __getitem__(self, index):
        """
        Override prev implementation to help with filtered metrics

        Add dimension to triplet to indicate whether it is a true triplet or not.

        True is denoted by 0 and False by 1
        """
        corrupted_head_triplets = []
        corrupted_tail_triplets = []
        head, relation, tail = self.triplets[index]

        for e in range(self.num_entities):
            corrupt_head = (e, relation, tail)
            corrupt_tail = (head, relation, e)

            corrupt_head_bit = (e, relation, tail) != self.triplets[index]
            corrupt_tail_bit = (e, relation, tail) != self.triplets[index]
            # corrupt_head_bit = int(self.all_triplets.get(corrupt_head) is None)
            # corrupt_tail_bit = int(self.all_triplets.get(corrupt_tail) is None)

            corrupted_head_triplets.append(corrupt_head + (corrupt_head_bit,))
            corrupted_tail_triplets.append(corrupt_tail + (corrupt_tail_bit,))

        # Why am i doing this?
        random.shuffle(corrupted_head_triplets)
        random.shuffle(corrupted_tail_triplets)

        return torch.LongTensor(self.triplets[index]), torch.LongTensor(corrupted_head_triplets), torch.LongTensor(corrupted_tail_triplets)



class AllDataSet():
    """
    Base class for all possible datasets
    """

    def __init__(self, dataset_name, relation_pos="middle"):
        self.dataset_name = dataset_name
        self.relation_pos = relation_pos

        self.entity2idx, self.relation2idx = self._load_mapping()
        self.entities, self.relations = list(set(self.entity2idx)), list(set(self.relation2idx))

        self.train_triplets = self._load_triplets("train")
        self.valid_triplets = self._load_triplets("valid")
        self.test_triplets = self._load_triplets("test")


    @property
    def num_entities(self):
        return len(self.entities)
    
    @property
    def num_relations(self):
        return len(self.relations)

    @property
    def all_triplets(self):
        return list(set(self.train_triplets + self.valid_triplets + self.test_triplets))
    

    def __getitem__(self, key):
        """
        Get specific dataset
        """
        if key == 'train':
            return self.train
        if key == 'validation':
            return self.validation
        if key == "test":
            return self.test

        print("No key with name", key)

        return None


    def _load_mapping(self):
        """
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

        Args:
            data_split: Which split of the data to load (train/test/validation)

        Returns:
            list of tuples: Triplets
        """
        triplets = []

        with open(os.path.join(DATA_DIR, self.dataset_name, f"{data_split}.txt"), "r") as file:
            for line in file:
                fields = [l.strip() for l in line.split()]

                if self.relation_pos.lower() == "middle":
                    triplets.append((self.entity2idx[fields[0]], self.relation2idx[fields[1]], self.entity2idx[fields[2]]))
                else:
                    triplets.append((self.entity2idx[fields[0]], self.relation2idx[fields[2]], self.entity2idx[fields[1]]))

        return triplets




#######################################################
#######################################################
#######################################################


class FB15k_237(AllDataSet):
    """
    Load the FB15k-237 dataset
    """
    def __init__(self):
        super().__init__("FB15k-237")


class WN18RR(AllDataSet):
    """
    Load the WN18RR dataset
    """
    def __init__(self):
        super().__init__("WN18RR")



class FB15k(AllDataSet):
    """
    Load the FB15k dataset
    """
    def __init__(self):
        super().__init__("FB15k", relation_pos="end")


class WN18(AllDataSet):
    """
    Load the WN18 dataset
    """
    def __init__(self):
        super().__init__("WN18", relation_pos="end")

