# %%
import torch
from datasets import load_dataset, Dataset
from functools import partial
import os
from tqdm import tqdm

# %%
imnet = load_dataset("imagenet-1k", split="train", streaming=True)
# %%
already_saved = 100000
imnet = imnet.skip(already_saved)
imnet_subset = imnet.take(100000)
# %%
def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

ds = Dataset.from_generator(partial(gen_from_iterable_dataset, imnet_subset), features=imnet_subset.features)

# %%
len(list(set(ds['label'])))
# %%
# ds['image']
# ds['label']

for i in tqdm(range(len(ds))):
    ds[i]['image'].save(f"imnet/train/class{ds[i]['label']}/img{already_saved + i}.jpeg")

# %%
