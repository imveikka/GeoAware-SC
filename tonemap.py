import numpy as np    
import cv2
import glob
import argparse
import os
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Tonemap using Mantiuk's method")

    # Add arguments
    parser.add_argument('-i', '--images', help='path to image file(s)')
    parser.add_argument('-d', '--dest', help='output directory')
    parser.add_argument('-g', '--gamma', help='gamma', type=float, default=1.0)
    parser.add_argument('-f', '--scale', help='scale', type=float, default=0.7)
    parser.add_argument('-s', '--saturation', help='saturation', type=float, default=1.0)

    # Parse the arguments
    args = parser.parse_args()   

    # Init paths
    all_files = glob.glob(args.images)
    os.makedirs(args.dest, exist_ok=True)

    # Init tonemapper
    tonemapper = cv2.createTonemapMantiuk(gamma=args.gamma, scale=args.scale, saturation=args.saturation)

    # Process all
    for file in tqdm(all_files):
        img = cv2.imread(file)
        img = img.astype('float32') / 255
        img = tonemapper.process(img)
        img[np.where(np.isnan(img))] = 0
        img = np.clip(img * 255, 0, 255).astype('uint8')
        out = file.rsplit('/', 1)[-1]
        cv2.imwrite(os.path.join(args.dest, out), img)
