# %%

import torch
from models.lenet_edit import GoogLeNet
import pickle
from torch.utils.data import DataLoader, Subset
import numpy as np
import glob
# %%
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
model.eval()

# %%

model.state_dict()
# %%
# from torchvision.models import GoogLeNet
# # %%
# # %%
# GoogLeNet()

# %%
my_model = GoogLeNet(aux_logits=False)
my_model.load_state_dict(model.state_dict())
my_model.eval()
model = my_model

# %%

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
        with open(f"lenet_results/plausibility_loss_all_neurons_{n_classes}.pkl", "wb") as f:
            pickle.dump(pl, f)
        with open(f"lenet_results/alt_loss_all_neurons_{n_classes}.pkl", "wb") as f:
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

