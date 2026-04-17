import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, rgb_to_grayscale, crop, pil_to_tensor
from PIL import Image
from kornia.morphology import opening, closing
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from utils.utils_correspondence import resize
from scipy.spatial import Delaunay
import networkx as nx
from torch_geometric.utils import from_networkx
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class Matches:
    kp_query: np.ndarray
    kp_database: np.ndarray
    distances: np.ndarray


def get_feature_map(img_path):
    
    desc_path = img_path.replace('.jpg', '.pt')
    desc = torch.load(desc_path)
    return desc.squeeze(0)


def create_circular_mask(s, center=None, radius=None):

    h, w = s, s
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask.astype(float)


def get_mask(img_path, device='cpu'):

    # img_path_seg = img_path.replace('source', 'segmented')
    # img = Image.open(img_path_seg)
    # img = resize(img, target_res=max(img.size))
    # img = to_tensor(img).to(device=device)
# 
    # mask = (img > 0).any(0).unsqueeze(0).unsqueeze(0).float()
    # kernel = torch.tensor(create_circular_mask(15)).float().to(device=device)
    # mask = opening(mask, kernel)
    # mask = closing(mask, kernel)
# 
    # return mask.squeeze(0).squeeze(0).cpu()
    img_path_seg = img_path.replace('.jpg', '_feature_mask.pt')
    return torch.load(img_path_seg)


def mask_to_feature_map(mask):
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(mask, 60, mode='nearest')
    return mask.squeeze(0).squeeze(0)


def prepare_image(img_path):
    desc = get_feature_map(img_path)
    mask = get_mask(img_path).to(desc.device)
    return desc, mask


def extract_features(desc, mask, return_idx=True):
    idx = torch.where(mask.bool())
    features = desc.permute(1, 2, 0)
    features = features[idx]
    if return_idx:
        return features, idx
    else:
        return features


def visualize_pair(img1_path, img2_path, figsize=(10, 10)):

    desc1, mask1 = prepare_image(img1_path)
    desc2, mask2 = prepare_image(img2_path)
    torch.cuda.empty_cache()
    
    # mask1, mask2 = map(mask_to_feature_map, (mask1, mask2))
    f1, i1 = extract_features(desc1, mask1)
    f2, i2 = extract_features(desc2, mask2)
    l1 = f1.size(0)

    X = torch.cat((f1, f2)).cpu().double().numpy()
    colors = KernelPCA(n_components=3, kernel='rbf').fit_transform(X)
    colors = (colors - colors.min()) / (colors.max() - colors.min())

    c1 = colors[:l1]
    c2 = colors[l1:]

    seg1 = np.zeros((60, 60, 3))
    seg2 = np.zeros((60, 60, 3))

    i1 = tuple(map(lambda x: x.cpu().numpy(), i1))
    i2 = tuple(map(lambda x: x.cpu().numpy(), i2))

    seg1[i1] = c1
    seg2[i2] = c2

    img1 = resize(Image.open(img1_path), 480)
    img2 = resize(Image.open(img2_path), 480)

    fig, ax = plt.subplots(2, 2, figsize=figsize)

    ax[0, 0].imshow(img1)
    ax[0, 1].imshow(img2)
    ax[1, 0].imshow(seg1)
    ax[1, 1].imshow(seg2)

    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')


def visualize_matches(img1_path, img2_path, target_res=60,
                      num_matches=10, figsize=(10, 5)):

    img1, img2 = map(Image.open, (img1_path, img2_path))
    img1, img2 = map(lambda x: resize(x, 512), (img1, img2))
    img1, img2 = map(np.array, (img1, img2))

    map1, mask1 = prepare_image(img1_path)
    map2, mask2 = prepare_image(img2_path)

    # mask1, mask2 = map(mask_to_feature_map, (mask1, mask2))
    desc1, kp1 = extract_features(map1, mask1)
    desc2, kp2 = extract_features(map2, mask2)

    desc1 = desc1.cpu().numpy()
    desc2 = desc2.cpu().numpy()

    kp1 = torch.stack(kp1).cpu().double().numpy().T
    kp2 = torch.stack(kp2).cpu().double().numpy().T

    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)

    kp1 *= (img1.shape[0] / mask1.shape[0])
    kp2 *= (img2.shape[0] / mask2.shape[0])

    kp1 = [cv2.KeyPoint(x=float(x), y=float(y), size=1) for y, x in kp1]
    kp2 = [cv2.KeyPoint(x=float(x), y=float(y), size=1) for y, x in kp2]

    fig, ax = plt.subplots(figsize=figsize)
    image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:num_matches], 
                            None, matchColor=(0, 255, 0),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    ax.imshow(image)
    ax.axis('off')


def collect_patches(path, kp, img_size=600, desc_size=60):

    patch_size = img_size // desc_size
    target_size = 32
    shift = (target_size - patch_size) // 2

    idx_big = kp.mul(patch_size).sub(shift)

    img = resize(Image.open(path.replace('source', 'segmented')), img_size)
    img = pil_to_tensor(img)

    patches = [
        crop(img, i, j, target_size, target_size)
        for i, j in idx_big
    ]
    patches = torch.stack(patches)
    return patches


def delaunay_to_pyg(tri, patches):
    G = nx.Graph()
    for path in tri.simplices: 
        nx.add_cycle(G, path)
    pos = dict(enumerate(torch.tensor(tri.points)))
    patches = dict(enumerate(patches))
    nx.set_node_attributes(G, pos, 'yx')
    nx.set_node_attributes(G, patches, 'patch')
    return from_networkx(G)


def get_graphs(path_query, path_database, match):

    kp_query = torch.tensor(match.kp_query)
    kp_database = torch.tensor(match.kp_database)

    patch_query = collect_patches(path_query, kp_query)
    patch_database = collect_patches(path_database, kp_database)

    tri_query = Delaunay(kp_query)
    tri_database = Delaunay(kp_database)

    G_query = delaunay_to_pyg(tri_query, patch_query)
    G_database = delaunay_to_pyg(tri_database, patch_database)

    return {'query': G_query, 'database': G_database}


def _get_graphs(name, path_query, path_database, match):
    return {name: get_graphs(path_query, path_database, match)}


def inference_graphs(path_query, matches_dict, max_workers=1):

    path, name = path_query.rsplit('/', 1)
    name = name[:-4]
    databases = matches_dict[name].keys()
    paths_database = map(lambda x: path.replace('query', 'database') + f'/{x}.jpg', databases)
    arg_list = [(db, path_query, db_img, matches_dict[name][db]) \
                for db, db_img in zip(databases, paths_database)]

    graphs = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_get_graphs, *args) for args in arg_list]
        for future in futures:
            graphs.update(future.result())

    return graphs

