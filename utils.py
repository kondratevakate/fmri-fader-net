import os
import numpy as np
import torch
import pydicom
import nibabel as nib
import nibabel.processing
import matplotlib.pyplot as plt

# def load_nii_to_array(nii_path,voxel_size = [2, 2, 2]):
#     file_ = nib.load(nii_path)
#     resampled_img = nibabel.processing.resample_to_output(file_, voxel_size)
#     return resampled_img.get_data()

def load_nii_to_array(nii_path):
    return nib.load(nii_path).get_data()

def min_max_scale(x):
    return (x - x.min()) / (x.max() - x.min())

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)
        
def save_res(res, path):
    ensure_dir(path)
    with open(path, "w") as f:
        f.write(str(res))
        
def load_res(path):
    with open(path) as f:
        res = f.read()
    return eval(res.replace("nan", "np.nan"))

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    state - dict containing:
    "model" : model.state_dict(),
    "optimizer" : optimizer.state_dict(),
    (optionally) loss, epoch, etc.
    """
    ensure_dir(filename)
    torch.save(state, filename)
    
def load_checkpoint(filename):
    """
    state - dict containing:
    "model" : model.state_dict(),
    "optimizer" : optimizer.state_dict()
    """
    state = torch.load(filename)
    return state
    
# def load_checkpoint(filename):
#     """
#     """
# #     model = TheModelClass(*args, **kwargs)
# #     optimizer = TheOptimizerClass(*args, **kwargs)

#     checkpoint = torch.load(filename)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     opt.load_state_dict(checkpoint['optimizer_state_dict'])
# #     epoch = checkpoint['epoch']
# #     loss = checkpoint['loss']
    
    
def load_results(name, problem, metric="auc"):
    train_loss_l = load_res("models/{}/{}/train_loss".format(
        name, problem.replace("/", "_")))
    val_loss_l = load_res("models/{}/{}/val_loss".format(
        name, problem.replace("/", "_")))
    train_metric_l = load_res("models/{}/{}/train_{}".format(
        name, problem.replace("/", "_"), metric))
    val_metric_l = load_res("models/{}/{}/val_{}".format(
        name, problem.replace("/", "_"), metric))
#     val_last_preds_l = load_res("models/" + problem_name + "/val_last_probs_" + problem.replace("/", "_"))
    return train_loss_l, val_loss_l, train_metric_l, val_metric_l #, val_last_preds_l
    
def save_results(name, problem, 
                 train_loss_l=[], 
                 val_loss_l=[], 
                 train_metric_l=[], 
                 val_metric_l=[], 
                 val_last_preds_l=None,
                 metric="auc"):
    save_res(train_loss_l, "models/{}/{}/train_loss".format(
        name, problem.replace("/", "_")))
    save_res(val_loss_l, "models/{}/{}/val_loss".format(
        name, problem.replace("/", "_")))
    save_res(train_metric_l, "models/{}/{}/train_{}".format(
        name, problem.replace("/", "_"), metric))
    save_res(val_metric_l, "models/{}/{}/val_{}".format(
        name, problem.replace("/", "_"), metric))
    if val_last_preds_l is not None:
        raise NotImplementedError
    print("saved.")

    
def plot_losses(problem_name, problem, mean=False, metric="auc"):
    train_loss_l, val_loss_l, train_metric_l, val_metric_l = load_results(problem_name, problem, metric)
    if mean:
        plt.figure(figsize=(10, 5))
        plt.plot(np.mean(train_loss_l, axis=0))
        plt.plot(np.mean(val_loss_l, axis=0))
        plt.show()
    
    else:
        plt.figure(figsize=(30, 10))
        for i in range(len(train_loss_l)):
            plt.subplot(3, 5, i + 1)
            plt.plot(train_loss_l[i])
            plt.plot(val_loss_l[i])
        plt.show()
        
def plot_metrics(problem_name, problem, mean=False, metric="auc"):
    train_loss_l, val_loss_l, train_metric_l, val_metric_l = load_results(problem_name, problem, metric)
    if mean:
        plt.figure(figsize=(10, 5))
        plt.plot(np.mean(train_metric_l, axis=0))
        plt.plot(np.mean(val_metric_l, axis=0))
        plt.show()
    
    else:
        plt.figure(figsize=(30, 10))
        for i in range(len(train_loss_l)):
            plt.subplot(3, 5, i + 1)
            plt.plot(train_metric_l[i])
            plt.plot(val_metric_l[i])
            plt.ylim(0.0, 1.0)
        plt.show()
        
###
# updated utils functions for train & cross_val_train returning train_stats in a form of dict
# (see train_dann & cross_val_train_dann)
        
def save_stats(name, problem, stats_to_save):
    """
    Args:
    --- stats_to_save - (dict of lists) - all the collected stats to save(from cross_val_train).
    """
    for stat_name in stats_to_save:
        save_res(stats_to_save[stat_name], 
                 "models/{}/{}/{}".format(
                     name, problem.replace("/", "_"), stat_name))
    print("saved.")


def load_stats(name, problem, stats_to_load):
    """
    Args:
    --- stats_to_load - (dict of lists) - an (empty) dict with keys corresponding to the stats to load.
    Returns:
    --- filled stats_to_load (but also fills them in-place)
    """
    for stat_name in stats_to_load:
        stats_to_load[stat_name] = load_res(
            "models/{}/{}/{}".format(
                name, problem.replace("/", "_"), stat_name))
    return stats_to_load


def plot_stats(problem_name, problem, stats_to_load, 
               stats_to_plot, mean=False, ylims=None):
    stats_to_load = load_stats(problem_name, problem, stats_to_load)
    if mean:
        plt.figure(figsize=(10, 5))
        for stat_name in stats_to_load:
            if stat_name in stats_to_plot:
                stat_vals = np.mean(stats_to_load[stat_name], axis=0)
                plt.plot(stat_vals, label=stat_name)
                if ylims is not None:
                    plt.ylim(*ylims)
        plt.legend()
        plt.show()
    
    else:
        plt.figure(figsize=(30, 10))
        for stat_name in stats_to_load:
            if stat_name in stats_to_plot:
                stat_vals = stats_to_load[stat_name]
                for i in range(len(stat_vals)):
                    plt.subplot(3, 5, i + 1)
                    plt.plot(stat_vals[i], label=stat_name)
                    if ylims is not None:
                        plt.ylim(*ylims)
        plt.legend()
        plt.show()