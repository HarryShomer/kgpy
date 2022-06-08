"""
Create entity2id.txt and relation2id.txt files for each CoDEx dataset
"""
import os 

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "datasets")


def load_data(dataset_name):
    """
    Load all unique entities + relations

    Parameters:
    -----------
        dataset_name: str 
            Which codex dataset we are using

    Returns:
    --------
    tuple
        entities, relations
    """
    ents, rels = set(), set()

    for split in ['train', 'valid', 'test']:
        with open(os.path.join(DATA_DIR, dataset_name, f"{split}.txt"), "r") as file:
            for line in file:
                fields = [l.strip() for l in line.split()]

                rels.add(fields[1])
                ents.update([fields[0], fields[2]])
            
    return ents, rels


def save_ents_rels(unique_ents, unique_rels, dataset_name):
    """
    Save unique entities and relations for given CoDEx dataset size

    Save in {entity/relation}2id.txt format for both

    Parameters:
    -----------
        unique_ents: set
            All unique entities in dataset
        unique_rels: set
            All unique relations in dataset
        dataset_name: str 
            Which codex dataset we are using
    
    Returns:
    --------
    None
    """
    print("Writing Entities + Relations to File...")

    with open(os.path.join(DATA_DIR, dataset_name, f"entity2id.txt"), "w") as file:
        for ix, ent in enumerate(unique_ents):
            file.write(f"{ent}    {ix}\n")

    with open(os.path.join(DATA_DIR, dataset_name, f"relation2id.txt"), "w") as file:
        for ix, rel in enumerate(unique_rels):
            file.write(f"{rel}    {ix}\n")


def main():
    for codex_size in ["S", "M", "L"]:
        print(f"Processing CoDEx-{codex_size}")

        ents, rels = load_data(f"CODEX-{codex_size}")
        save_ents_rels(ents, rels, f"CODEX-{codex_size}")


if __name__ == "__main__":
    main()
