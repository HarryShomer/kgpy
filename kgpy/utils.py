import os
import torch
from random import randint, choice



def save_model(model, optimizer, epoch, step, dataset_name, checkpoint_dir, suffix=""):
    """
    Save the given model's state
    """
    suffix = "_" + str(suffix) if len(str(suffix)) > 0 else str(suffix)

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.isdir(os.path.join(checkpoint_dir, dataset_name)):
        os.mkdir(os.path.join(checkpoint_dir, dataset_name))

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "latent_dim": model.dim,
        "loss_fn": model.loss_fn_name,
        "regularization": model.regularization,
        "reg_weight": model.reg_weight,
        "epoch": epoch,
        "step": step
    }, os.path.join(checkpoint_dir, dataset_name, f"{model.name}{suffix}.tar"))


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
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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