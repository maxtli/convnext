# %%
import torch
import pickle
import seaborn as sns
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
# %%

def gauss_kernel(ts,sigma):
    return torch.exp(-1 * torch.square(ts) / (2 * sigma * sigma))

# %%
def get_importance(loss, alt_loss, sigma_low=1e-4, sigma_high=1e-2, step=1e-4):
    sigmas = []
    imp_tensors = []
    tpl = []
    for i in tqdm(range(int((sigma_high - sigma_low) // step))):
        sigma = sigma_low + i * step
        p_weights = gauss_kernel(loss,sigma)
        imp_tensors.append(torch.sum(p_weights * alt_loss, dim=0) / torch.sum(p_weights, dim=0))
        sigmas.append(sigma)
        tpl.append(torch.sum(p_weights,dim=0))
    
    max_impt = [torch.max(x).item() for x in imp_tensors]
    mean_impt = [torch.median(x).item() for x in imp_tensors]
    std_impt = [torch.quantile(x,.75).item() for x in imp_tensors]
    std_low_impt = [torch.quantile(x,.25).item() for x in imp_tensors]
    return sigmas, imp_tensors, max_impt, mean_impt, std_impt, std_low_impt, tpl

# %%

def get_importances_for_dims(i):
    with open(f"results/plausibility_loss_all_neurons_{i}.pkl", "rb") as f:
        plausibility_losses = pickle.load(f)
    with open(f"results/alt_loss_all_neurons_{i}.pkl", "rb") as f:
        alt_losses = pickle.load(f)

    plausibility_losses = torch.cat(plausibility_losses, dim=0)
    alt_losses = torch.cat(alt_losses, dim=0)

    return get_importance(plausibility_losses, alt_losses), plausibility_losses.shape


# %%
for dim in [3,5,7,10,15,20,30,35]:
    plt.figure()
    plt.title(f"Neuron importances with {dim} output dimensions")
    (sigmas, ts, max, mean, stdp, stdm, tpl), shp = get_importances_for_dims(dim)
    ax1=sns.lineplot(x=sigmas, y=max, label="max")
    sns.lineplot(x=sigmas, y=mean, label="median")
    sns.lineplot(x=sigmas, y=stdp, label="75th")
    sns.lineplot(x=sigmas, y=stdm, label="25th")

    max_samp = [torch.max(x).item() / shp[0] for x in tpl]
    mean_samp = [torch.median(x).item() / shp[0] for x in tpl]
    std_samp = [torch.quantile(x,.75).item() / shp[0] for x in tpl]
    std_low_samp = [torch.quantile(x,.25).item() / shp[0] for x in tpl]

    ax1=sns.lineplot(x=sigmas, y=max_samp, label="max", ax=ax1.twinx())
    sns.lineplot(x=sigmas, y=mean_samp, label="median")
    sns.lineplot(x=sigmas, y=std_samp, label="75th")
    sns.lineplot(x=sigmas, y=std_low_samp, label="25th")

    ax1.legend()
    plt.savefig(f"results/fig_{dim}.png")

    layers = [96,192,384,768]
    start_idx = 0
    for i,layer_dim in enumerate(layers):
        plt.figure()
        plt.title(f"Neuron importances after layer {i} with {dim} output dimensions")
        layer_ts = [x[start_idx:start_idx+layer_dim] for x in ts]
        layer_tpl = [x[start_idx:start_idx+layer_dim] for x in tpl]
        max_impt = [torch.max(x).item() for x in layer_ts]
        mean_impt = [torch.median(x).item() for x in layer_ts]
        std_impt = [torch.quantile(x,.75).item() for x in layer_ts]
        std_low_impt = [torch.quantile(x,.25).item() for x in layer_ts]
        ax1=sns.lineplot(x=sigmas, y=max_impt, label="max")
        sns.lineplot(x=sigmas, y=mean_impt, label="median")
        sns.lineplot(x=sigmas, y=std_impt, label="75th")
        sns.lineplot(x=sigmas, y=std_low_impt, label="25th")

        max_samp = [torch.max(x).item() / shp[0] for x in layer_tpl]
        mean_samp = [torch.median(x).item() / shp[0] for x in layer_tpl]
        std_samp = [torch.quantile(x,.75).item() / shp[0] for x in layer_tpl]
        std_low_samp = [torch.quantile(x,.25).item() / shp[0] for x in layer_tpl]

        ax1=sns.lineplot(x=sigmas, y=max_samp, label="max", ax=ax1.twinx())
        sns.lineplot(x=sigmas, y=mean_samp, label="median")
        sns.lineplot(x=sigmas, y=std_samp, label="75th")
        sns.lineplot(x=sigmas, y=std_low_samp, label="25th")
        ax1.legend()

        plt.savefig(f"results/fig_{dim}_layer_{i}.png")
        start_idx += layer_dim


# %%
for dim in [3,5,7,10,15,20]:
    (sigmas, ts, max, mean, stdp, stdm, tpl), shp = get_importances_for_dims(dim)

    layers = [96,192,384,768]
    start_idx = 0
    for i,layer_dim in enumerate(layers):
        plt.figure()
        plt.title(f"Neuron importances self-correlation after layer {i} with {dim} output dimensions")
        layer_ts = [x[start_idx:start_idx+layer_dim] for x in ts]
        print(layer_ts[0].shape)
        sns.scatterplot(x=(layer_ts[5]).numpy(), y=(layer_ts[10]).numpy(), s=5)
        plt.savefig(f"results/self-corr/fig_{dim}_layer_{i}.png")



# %%

# %%
ts[0].shape
# %%
