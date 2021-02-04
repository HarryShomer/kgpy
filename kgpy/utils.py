import os
import torch

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


def load_model(model, optimizer, epoch, dataset_name):
    """
    """
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, dataset_name, f"{model.name}_epoch_{epoch}.tar"))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizersstate_dict'])

    return model, optimizer