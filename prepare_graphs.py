from utils.my_stuff import *
from glob import glob
import pickle
from multiprocessing import Pool
from functools import partial
import torch
from tqdm import tqdm


# --- Change ---
query_imgs = sorted(glob('../data/SealID/full images/source_query_tn/*.jpg'))
database_imgs = sorted(glob('../data/SealID/full images/source_database_tn/*.jpg'))
# --------------

with open('matches.pkl', 'rb') as buf:
    matches = pickle.load(buf)

for query_path in tqdm(query_imgs[1600:]):

    name = query_path.rsplit('/', 1)[-1][:-4]
    graphbook = inference_graphs(query_path, matches, 4)

    # --- Change --- (create the directory in advance if necessary)
    torch.save(graphbook, f'../data/SealID/graphs/graphbook_{name}.pt')
    # --------------