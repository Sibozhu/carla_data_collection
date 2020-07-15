import argparse

import cv2
import os
import numpy as np
from tqdm import tqdm

def semantic_processor(dataset_uri):

    
    for image_path in tqdm(os.listdir(dataset_uri)):

        image = os.path.join(dataset_uri,image_path)

        output_path_bin = image.split('/')[-4] + "/" + image.split('/')[-3] + "/" + "processed_2d_seg/" + image.split('/')[-1].split('.')[0] + ".bin"

        if not os.path.exists(image.split('/')[-4] + "/" + image.split('/')[-3] + "/" + "processed_2d_seg/"):
            os.makedirs(image.split('/')[-4] + "/" + image.split('/')[-3] + "/" + "processed_2d_seg/")

        seg_matrix = cv2.imread(image)

        seg_matrix = np.delete(seg_matrix,1,2)
        seg_matrix = np.delete(seg_matrix,0,2)
        seg_matrix = seg_matrix.squeeze()
        final_list = seg_matrix.flatten()

        # save original prediction as bin file
        f = open(output_path_bin, 'w+b')
        f.write(final_list)
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})    

    parser.add_argument("--dataset_uri", type=str, help="path to dataset root folder", default="images/2020-07-14_22_20/raw_semantic")
    arg = parser.parse_args()  
    semantic_processor(
        dataset_uri=arg.dataset_uri
        )