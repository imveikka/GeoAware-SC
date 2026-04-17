from utils.my_stuff import *
from glob import glob
from cv2 import BFMatcher
import pandas as pd
from tqdm import tqdm
import pickle
from dataclasses import dataclass
import numpy as np
import torch
from itertools import product


@dataclass
class Matches:
    kp_query: np.ndarray
    kp_database: np.ndarray
    distances: np.ndarray


def expand_matches(matches):
    q_idx, d_idx, dist = np.array([[m.queryIdx, m.trainIdx, m.distance] for m in matches]).T
    return q_idx.astype(int), d_idx.astype(int), dist


def get_metadata(path):
    name = path.rsplit('/', 1)[-1][:-4]
    desc, mask = map(lambda x: x.cpu(), prepare_image(path))
    # mask = mask_to_feature_map(mask)
    feat, idx = extract_features(desc, mask)
    feat = feat.numpy()
    kp = torch.stack(idx).numpy().T
    return {
        'name': name,
        'feat': feat,
        'kp': kp
    }


if __name__ == '__main__':

    matcher = BFMatcher(crossCheck=True)

    query_imgs = sorted(glob('../data/SealID/full images/source_query_tn/*.jpg'))
    database_imgs = sorted(glob('../data/SealID/full images/source_database_tn/*.jpg'))

    db_titles = [path.rsplit('/', 1)[-1][:-4] for path in database_imgs]
    qr_titles = [path.rsplit('/', 1)[-1][:-4] for path in query_imgs]
    df = {name: {} for name in qr_titles}

    matcher = cv2.BFMatcher(crossCheck=True)

    query = list(map(get_metadata, tqdm(query_imgs)))
    database = list(map(get_metadata, tqdm(database_imgs)))

    for qr, db in tqdm(product(query, database), total=len(query_imgs)*len(database_imgs)):

        name_q = qr['name']
        feat_q = qr['feat']
        kp_q = qr['kp']

        name_d = db['name']
        feat_d = db['feat']
        kp_d = db['kp']
        
        matches = matcher.match(feat_q, feat_d)
        idx_q, idx_d, dists = expand_matches(matches)
        kp_q = kp_q[idx_q]
        kp_d = kp_d[idx_d]

        order = np.argsort(dists)

        df[name_q][name_d] = Matches(
            kp_query=kp_q[order],
            kp_database=kp_d[order],
            distances=dists[order]
        )

    with open('matches.pkl', 'wb') as buf:
        pickle.dump(df, buf)