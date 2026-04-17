from utils.my_stuff import *
from PIL import Image
import torch
from kornia.morphology import opening, closing
import sys
import os
from tqdm import tqdm


def save_mask(img_path, device='cpu'):

    img_path_seg = img_path.replace('source', 'segmented')
    img = Image.open(img_path_seg)
    img = resize(img, target_res=max(img.size))
    img = to_tensor(img).to(device=device)

    mask = (img > 0).any(0).unsqueeze(0).unsqueeze(0).float()
    kernel = torch.tensor(create_circular_mask(15)).float().to(device=device)
    mask = opening(mask, kernel)
    mask = closing(mask, kernel).to('cpu')
    mask = F.interpolate(mask, 60, mode='nearest')

    img_path_seg = img_path.replace('.jpg', '_feature_mask.pt')
    return torch.save(mask.squeeze(0).squeeze(0), img_path_seg)


if __name__ == '__main__':

    assert len(sys.argv) > 1, "Give the base directory!"

    base_dir = sys.argv[1]
    all_files = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(base_dir) for file in files if file.endswith('.jpg')]

    for path in tqdm(all_files):
        save_mask(path, device='cuda')