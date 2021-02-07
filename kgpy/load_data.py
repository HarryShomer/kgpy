import os
import json
import numpy as np 
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "kg_datasets")



class CustomDataset(torch.utils.data.Dataset):
    """
    Extend torch.utils.data.Dataset.

    Loads a given split for a data set (e.g. training data)
    """

    def __init__(self, dataset_name, data_split, entity2idx, relation2idx, relation_pos="middle"):
        self.dataset_name = dataset_name
        self.data_split = data_split

        self.entity2idx, self.relation2idx = entity2idx, relation2idx
        self.triplets = self._load_triplets(data_split, relation_pos)

        #print(len(entity2idx), len(relation2idx), len(self.triplets))


    def __len__(self):
        """
        Length of dataset
        """
        return len(self.triplets)


    def __getitem__(self, index):
        """
        Get indicies for the ith triplet
        """
        t = self.triplets[index]

        return int(self.entity2idx[t[0]]), int(self.relation2idx[t[1]]), int(self.entity2idx[t[2]])


    def _load_triplets(self, data_split, triplet_order):
        """
        Load the triplets for a given dataset and data split.

        Args:
            data_split: Which split of the data to load (train/test/validation)
            relation_pos: Where the relation is one each line. Either "middle" or "end"

        Returns:
            list: Triplets
        """
        triplets = []

        with open(os.path.join(DATA_DIR, self.dataset_name, f"{data_split}.txt"), "r") as file:
            for line in file:
                line_components = [l.strip() for l in line.split()]

                if triplet_order.lower() == "middle":
                    triplets.append(line_components)
                else:
                    triplets.append([line_components[0], line_components[2], line_components[1]])

        return triplets


class AllDataSet():
    """
    Base class for all possible datasets
    """

    def __init__(self, dataset_name, relation_pos="middle"):
        self.dataset_name = dataset_name

        self.entity2idx, self.relation2idx = self._load_mapping()
        self.entities, self.relations = list(set(self.entity2idx)), list(set(self.relation2idx))

        self.train = CustomDataset(dataset_name, "train", self.entity2idx, self.relation2idx, relation_pos)
        self.validation = CustomDataset(dataset_name, "valid", self.entity2idx, self.relation2idx, relation_pos)
        self.test = CustomDataset(dataset_name, "test", self.entity2idx, self.relation2idx, relation_pos)


    @property
    def num_entities(self):
        return len(self.entities)
    
    @property
    def num_relations(self):
        return len(self.relations)


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
                entity2idx[line_components[0]] = line_components[1]

        with open(os.path.join(DATA_DIR, self.dataset_name, "relation2id.txt"), "r") as f:
            for line in f:
                line_components = [l.strip() for l in line.split()]
                relation2idx[line_components[0]] = line_components[1]

        return entity2idx, relation2idx


    def all_triplets(self):
        """
        Get all the index triplets across all the sets

        Returns:
            List of triplets
        """
        trip = [self.train[i] for i in range(len(self.train))]
        trip += [self.validation[i] for i in range(len(self.validation))]
        trip += [self.test[i] for i in range(len(self.test))]

        return trip


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

