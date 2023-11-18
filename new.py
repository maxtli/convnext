# %%
from main import get_args_parser
import argparse
from data import build_dataset
import torch
from torch.utils.data import DataLoader, Subset
import utils
import numpy as np
from timm.models import create_model
from models.convnext_edit import ConvNeXt, convnext_tiny
import json
import glob
from itertools import cycle
from tqdm import tqdm
import pickle
# %%
device = "cuda:0" if torch.cuda.is_available else "cpu"
parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args(["--data_path", "imnet", "--data_set", "IMNET", "--model", "convnext_tiny"])
# if args.output_dir:
#     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
# print(args)
# %%
dataset_train, nb_classes = build_dataset(True, args)

# num_tasks = utils.get_world_size()
# global_rank = utils.get_rank()

# sampler_train = .DistributedSampler(
#     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
# )
# data_loader_train = DataLoader(
#     dataset_train, sampler=sampler_train,
#     batch_size=args.batch_size,
#     num_workers=args.num_workers,
#     pin_memory=args.pin_mem,
#     drop_last=True,
# )
# model = create_model(
#     args.model, 
#     pretrained=True, 
#     num_classes=args.nb_classes, 
#     drop_path_rate=args.drop_path,
#     layer_scale_init_value=args.layer_scale_init_value,
#     head_init_scale=args.head_init_scale,
# )
# %%
# cnn_model = convnext_tiny(pretrained=True)

# %%

# def save_model(model, path):
#     torch.save(model.state_dict(), path)

# convnext-tiny
def load_model_local(path):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    model.load_state_dict(torch.load(path))
    return model

# %%
path = "models/convnext_tiny.pth"
# save_model(cnn_model,path)

# %%
# cnn_tiny = load_model_local(path)

model = convnext_tiny(True).to(device)
# %%

# %%
bsz=8

# %%


correction_mapping = [int(j) for j in sorted([str(i) for i in range(1000)])]

#  the dataset is the model coordinates in ALPHABETICAL order

# %%
def run_plausibility(n_classes, record_every=20):
    max_item = np.sum([len(glob.glob(f"imnet/train/class{i}/*.jpeg")) for i in range(n_classes)])
    # data_loader_train = DataLoader(dataset_train, batch_size=bsz)
    # alt_loader = DataLoader(dataset_train, batch_size=bsz, shuffle=True)
    data_loader_train = DataLoader(Subset(dataset_train, range(0, max_item)), batch_size=bsz)
    alt_loader = DataLoader(Subset(dataset_train, range(0,max_item)), batch_size=bsz, shuffle=True)
    # subset_data = Subset(dataset_train, range(0, len(dataset_train), 20))
    # subset_loader = DataLoader(subset_data, batch_size=30)

    alt_cycle = cycle(alt_loader)
    criterion = torch.nn.CrossEntropyLoss()

    plausibility_losses = []
    alt_losses = []

    def save_losses(pl, alt): 
        with open(f"results/plausibility_loss_all_neurons_{n_classes}.pkl", "wb") as f:
            pickle.dump(pl, f)
        with open(f"results/alt_loss_all_neurons_{n_classes}.pkl", "wb") as f:
            pickle.dump(alt, f)

    for i, (samples, targets) in tqdm(enumerate(data_loader_train)):
        if samples.shape[0] < bsz:
            save_losses(plausibility_losses, alt_losses)
            break

        with torch.no_grad():
            alt_input = next(alt_cycle)
            alt_states = model(alt_input[0].to(device), partial="alt") 
            # print("alt")
            patched_res = model(samples.to(device), partial="primary", alts=alt_states)[...,correction_mapping[:n_classes]].softmax(dim=-1)
            # print("model")
            
            alt_probs = patched_res[:bsz].unsqueeze(1)
            orig_probs = patched_res[bsz:2*bsz].unsqueeze(1)
            patched_probs = patched_res[2*bsz:].reshape(orig_probs.shape[0], -1, orig_probs.shape[2])

            kl_loss = torch.sum(alt_probs * (alt_probs.log() - patched_probs.log()), dim=-1).cpu()
            kl_alt_loss = torch.sum(orig_probs * (orig_probs.log() - alt_probs.log()), dim=-1).cpu()

            plausibility_losses.append(kl_loss)
            alt_losses.append(kl_alt_loss)
        
        if i % record_every == record_every - 1:
            save_losses(plausibility_losses, alt_losses)
        
        # if correction_mapping is None:
        #     correction_mapping = torch.stack([logits.argmax(dim=-1), targets], dim=1)
        # else:
        #     correction_mapping = torch.cat([correction_mapping, torch.stack([logits.argmax(dim=-1), targets], dim=1)], dim=0)

# %%

test_dims = [5,7,10,15,20,25,30,35,40]
for dim in test_dims:
    run_plausibility(dim)

# %%
# plausibility_losses = torch.cat(plausibility_losses, dim=0)
# alt_losses = torch.cat(alt_losses, dim=0)

# # %%
# cor_map = sorted(list(str(i) for i in range(1000)))

# # %%

# with open("pl.pkl", "wb") as f:
#     pickle.dump(plausibility_losses.cpu(), f)
# with open("alt_l.pkl", "wb") as f:
#     pickle.dump(alt_losses.cpu(), f)
# %%
