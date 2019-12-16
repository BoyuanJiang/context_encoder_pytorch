import os
import sys
import glob
import random

MICCAI_PATH = '/home/dl/dataset/MICCAI2018_EndoVis_Challenge'
LABELED = ['train_data']


# LABELED = ['test_data']

def grab_img_gt_pair_paths(base_dir):
    image_paths, label_paths = [], []

    sequences = os.listdir(base_dir)

    for sequence in sequences:
        seq_dir = os.path.join(base_dir, sequence)
        if not os.path.isdir(seq_dir):
            continue

        image_dir = os.path.join(seq_dir, 'left_frames')
        # label_dir = os.path.join(seq_dir, 'labels')

        images = os.listdir(image_dir)
        for image in images:
            if '_' in image:
                continue

            image_paths.append(os.path.join(image_dir, image))
            # label_paths.append(os.path.join(label_dir, image))

    return image_paths, label_paths


if __name__ == '__main__':
    image_paths, label_paths = [], []
    for labeled_dir in LABELED:
        print(labeled_dir)
        _image_paths, _label_paths = grab_img_gt_pair_paths(os.path.join(MICCAI_PATH, labeled_dir))
        image_paths += _image_paths
        label_paths += _label_paths
        print(len(image_paths))

    random.shuffle(image_paths)

    with open('/home/flz/data_context/train_image.txt', 'w') as f:
        # for path in image_paths:
        for path in image_paths:
            f.write("%s\n" % path)

    with open('/home/flz/data_context/train_label.txt', 'w') as f:
        # for path in image_paths:
        for path in image_paths:
            f.write("%s\n" % path)

