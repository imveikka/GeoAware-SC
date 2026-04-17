import torch
import torch.nn.functional as F
from PIL import Image
from utils.utils_correspondence import resize
from utils.my_stuff import *
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
from kornia.morphology import opening, closing
import sys
import os
import numpy as np
from tqdm import tqdm


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_mask(img_path, device='cuda'):

    img_path_seg = img_path.replace('source', 'segmented')
    img = Image.open(img_path_seg)
    img = resize(img, target_res=max(img.size))
    img = to_tensor(img).to(device=device)

    mask = (img > 0).any(0).unsqueeze(0).unsqueeze(0).float()
    kernel = torch.tensor(create_circular_mask(15)).float().to(device=device)
    mask = opening(mask, kernel)
    mask = closing(mask, kernel)

    return mask.squeeze(0).squeeze(0).cpu()


def mask_to_feature_map(mask):
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(mask, 60, mode='nearest')
    return mask.squeeze(0).squeeze(0)


def sparsify(feature_map, mask_orig):
    dim = feature_map.shape[-2:]
    mask = F.interpolate(mask_orig, size=dim, mode='nearest')[0, 0]
    y, x = torch.where(~mask.bool())
    feature_map[0, :, y, x] = 0
    return feature_map.to_sparse()


@torch.inference_mode()
def save_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img_path):

    img = Image.open(img_path)

    img_mask = resize(img, target_res=max(img.size), resize=True, to_pil=True)
    img_mask = to_tensor(img_mask).unsqueeze(0).cuda()
    kernel_open = torch.tensor(create_circular_mask(5)).float().cuda()
    kernel_close = torch.tensor(create_circular_mask(9)).float().cuda()
    img_open = opening(img_mask, kernel_open)
    mask = (img_open.norm(dim=1, keepdim=True) > 0).float()
    mask = closing(mask, kernel_close)

    desc = {}

    # extract stable diffusion features
    img_sd_input = resize(img, target_res=num_patches*16, resize=True, to_pil=True)
    features_sd = process_features_and_mask(sd_model, sd_aug, img_sd_input, mask=False, raw=True)
    desc["sd"] = {}
    for key in ("s2", "s3", "s4", "s5"):
        desc["sd"][key] = sparsify(features_sd[key].clone(), mask).cpu()

    # extract dinov2 features
    img_dino_input = resize(img, target_res=num_patches*14, resize=True, to_pil=True)
    img_batch = (extractor_vit.preprocess_pil(img_dino_input)).cuda()
    features_dino = extractor_vit.extract_descriptors(img_batch, layer=11, facet='token')
    features_dino = features_dino.permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)
    desc["dino"] = sparsify(features_dino.clone(), mask).cpu()

    # aggregate the features and apply post-processing
    features_gathered = torch.cat([
        features_sd['s3'],
        F.interpolate(features_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
        F.interpolate(features_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
        features_dino], dim=1)
    features_geo = aggre_net(features_gathered) # 1, 768, 60, 60
    desc["geo"] = sparsify(features_geo.clone(), mask).cpu()

    output_path = img_path.replace('.png', '.pt')
    torch.save(desc, output_path)


if __name__ == '__main__':

    assert len(sys.argv) > 1, "Give the base directory!"

    # load the pretrained weights
    num_patches = 60
    set_seed()
    sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=num_patches*16, num_timesteps=50)
    extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')
    aggre_net = AggregationNetwork(feature_dims=[640,1280,1280,768], projection_dim=768, device='cuda')
    aggre_net.load_pretrained_weights(torch.load('results_spair/best_856.PTH'))

    sd_model.eval()
    aggre_net.eval()

    base_dir = sys.argv[1]
    all_files = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(base_dir) for file in files if file.endswith('.png')]

    for path in tqdm(all_files):
        save_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, path)
