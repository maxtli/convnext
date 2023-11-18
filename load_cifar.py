# %%
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from functools import partial
from itertools import cycle
from models.convnext import ConvNeXt, convnext_tiny

# %%
cifar = load_dataset("cifar100", split="train")

# %%
cnn_model = convnext_tiny(pretrained=True)
# %%

def save_model(model, path):
    torch.save(model.state_dict(), path)

# convnext-tiny
def load_model_local(path):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    model.load_state_dict(torch.load(path))
    return model

# %%
path = "models/convnext_tiny.pth"
save_model(cnn_model,path)

# %%
cnn_tiny = load_model_local(path)

# %%
cifar_loader = DataLoader(cifar, 100, True)
# %%
for x, batch in cifar_loader:
    print(batch)
    break

# cnn_tiny(next(cycle(cifar_loader)))
# %%

cifar.save_to_disk("hi")
# %%
for x in cifar:
    print("hi")
# %%
