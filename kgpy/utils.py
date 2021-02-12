import os
import torch
from random import randint

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "checkpoints")


def save_model(model, optimizer, epoch, step, dataset_name, suffix=""):
    """
    """
    suffix = "_" + str(suffix) if len(str(suffix)) > 0 else str(suffix)

    if not os.path.isdir(os.path.join(CHECKPOINT_DIR, dataset_name)):
        os.mkdir(os.path.join(CHECKPOINT_DIR, dataset_name))

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizersstate_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step
    }, os.path.join(CHECKPOINT_DIR, dataset_name, f"{model.name}{suffix}.tar"))


def load_model(model, optimizer, dataset_name, epoch=None):
    """
    Load the saved model
    """
    if epoch is None:
        file_path = os.path.join(CHECKPOINT_DIR, dataset_name, f"{model.name}.tar")
    else:
        file_path = os.path.join(CHECKPOINT_DIR, dataset_name, f"{model.name}_epoch_{epoch}.tar")

    if not os.path.isfile(file_path):
        print(f"The model checkpoint for {model.name} at epoch {epoch} was never saved.")
        return None, None

    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizersstate_dict'])

    return model, optimizer


def checkpoint_exists(model_name, dataset_name, epoch=None):
    """
    Check if a given checkpoint was ever saved
    """
    if epoch is None:
        file_path = os.path.join(CHECKPOINT_DIR, dataset_name, f"{model_name}.tar")
    else:
        file_path = os.path.join(CHECKPOINT_DIR, dataset_name, f"{model_name}_epoch_{epoch}.tar")

    return os.path.isfile(file_path)


def randint_exclude(begin, end, exclude):
    """
    Randint but exclude a number

    Args:
        begin: begin of range
        end: end of range (exclusive)
        exclude: number to exclude

    Returns:
        randint not in exclude
    """
    while True:
        x = randint(begin, end-1)

        if x != exclude:
            return x