import os
import gc
import sys
import torch
import warnings
from random import randint

def get_mem():
    """
    Print all params and memory usage

    **Used for debugging purposes**
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(obj.shape, type(obj), sys.getsizeof(obj.storage()))
        except: pass



class DataParallel(torch.nn.DataParallel):
    """
    Extend DataParallel class to access model level attributes/methods
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)



def save_model(model, optimizer, epoch, step, dataset_name, checkpoint_dir, suffix=""):
    """
    Save the given model's state
    """
    suffix = "_" + str(suffix) if len(str(suffix)) > 0 else str(suffix)

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.isdir(os.path.join(checkpoint_dir, dataset_name)):
        os.mkdir(os.path.join(checkpoint_dir, dataset_name))

    # If wrapped in DataParallel object this is how we access the underlying model
    if isinstance(model, DataParallel):
        model_obj = model.module
    else:
        model_obj = model

    torch.save({
        "model_state_dict": model_obj.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "latent_dim": model_obj.emb_dim,
        "loss_fn": model_obj.loss_fn.__class__.__name__,
        "regularization": model_obj.regularization,
        "reg_weight": model_obj.reg_weight,
        "epoch": epoch,
        "step": step
        }, 
        os.path.join(checkpoint_dir, dataset_name, f"{model.name}{suffix}.tar")
    )


def load_model(model, optimizer, dataset_name, checkpoint_dir, epoch=None):
    """
    Load the saved model
    """
    if epoch is None:
        file_path = os.path.join(checkpoint_dir, dataset_name, f"{model.name}.tar")
    else:
        file_path = os.path.join(checkpoint_dir, dataset_name, f"{model.name}_epoch_{epoch}.tar")

    if not os.path.isfile(file_path):
        print(f"The model checkpoint for {model.name} at epoch {epoch} was never saved.")
        return None, None


    # If wrapped in DataParallel object this is how we access the underlying model
    if isinstance(model, DataParallel):
        model_obj = model.module
    else:
        model_obj = model

    checkpoint = torch.load(file_path)
    model_obj.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizersstate_dict'])

    return model_obj, optimizer


def checkpoint_exists(model_name, dataset_name, checkpoint_dir, epoch=None):
    """
    Check if a given checkpoint was ever saved
    """
    if epoch is None:
        file_path = os.path.join(checkpoint_dir, dataset_name, f"{model_name}.tar")
    else:
        file_path = os.path.join(checkpoint_dir, dataset_name, f"{model_name}_epoch_{epoch}.tar")

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


