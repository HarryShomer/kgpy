import os
import json
import torch
import numpy as np 
from collections import defaultdict


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "kg_datasets")


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
        Number of triplets in training dataset
        """
        return len(self.triplets)


    def __getitem__(self, index):
        """
        Get indicies for the ith triplet -> head, relation, tail
        """
        return self.triplets[index][0], self.triplets[index][1], self.triplets[index][2]



class TestDataset(torch.utils.data.Dataset):
    """
    Dataset object for test data
    """

    def __init__(self, dataset_name, triplets, all_triplets, num_entities, evaluation_method):
        self.dataset_name = dataset_name
        self.triplets = triplets
        self.num_entities = num_entities
        self.evaluation_method = evaluation_method.lower()

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

            if self.evaluation_method == "filtered":
                corrupt_head_bit = int(self.all_triplets.get(corrupt_head) is None)
                corrupt_tail_bit = int(self.all_triplets.get(corrupt_tail) is None)
            else:
                corrupt_head_bit = (e, relation, tail) != self.triplets[index]
                corrupt_tail_bit = (head, relation, e) != self.triplets[index]

            corrupted_head_triplets.append(corrupt_head + (corrupt_head_bit,))
            corrupted_tail_triplets.append(corrupt_tail + (corrupt_tail_bit,))


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
            return self.train_triplets
        if key == 'validation':
            return self.valid_triplets
        if key == "test":
            return self.test_triplets

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

                if self.relation_pos.lower() == "middle":
                    triplets.append((self.entity2idx[fields[0]], self.relation2idx[fields[1]], self.entity2idx[fields[2]]))
                else:
                    triplets.append((self.entity2idx[fields[0]], self.relation2idx[fields[2]], self.entity2idx[fields[1]]))

        return triplets




#######################################################
#######################################################
#######################################################


class FB15K_237(AllDataSet):
    """
    Load the FB15k-237 dataset
    """
    def __init__(self):
        super().__init__("FB15K-237")


class WN18RR(AllDataSet):
    """
    Load the WN18RR dataset
    """
    def __init__(self):
        super().__init__("WN18RR")



class FB15K(AllDataSet):
    """
    Load the FB15k dataset
    """
    def __init__(self):
        super().__init__("FB15K", relation_pos="end")


class WN18(AllDataSet):
    """
    Load the WN18 dataset
    """
    def __init__(self):
        super().__init__("WN18", relation_pos="end")