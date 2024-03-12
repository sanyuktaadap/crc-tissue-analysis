# Libraries
import argparse
import os
import numpy as np
import math
import shutil
from tqdm import tqdm

def split_class(class_name, path_prefix, split_path_prefix, train_split, val_split):
    print(f"Splitting class - {class_name}")

    class_path = os.path.join(path_prefix, class_name)

    # Fetching all files inside this class.
    filenames = os.listdir(class_path)

    # Shuffling file names and splitting
    np.random.shuffle(filenames)
    train_len = math.floor(train_split * len(filenames))
    val_len = math.floor(val_split * len(filenames))
    train_names, val_names, test_names = (
        filenames[: train_len],
        filenames[train_len: train_len + val_len],
        filenames[train_len + val_len:]
    )

    # Helper function to copy
    def copy_file_to_split(name, split):
        src_file_path = os.path.join(class_path, name)
        dest_path = os.path.join(split_path_prefix, split, class_name)

        # Creating destination path folders if required.
        os.makedirs(dest_path, exist_ok=True)

        dest_file_path = os.path.join(dest_path, name)
        shutil.copyfile(src_file_path, dest_file_path)

    # Copying files to dedicated folder after splitting
    print("Copying train split")
    for name in tqdm(train_names):
        copy_file_to_split(name, split="train")

    print("Copying val split")
    for name in tqdm(val_names):
        copy_file_to_split(name, split="val")

    print("Copying test split")
    for name in tqdm(test_names):
        copy_file_to_split(name, split="test")


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='This script splits each class of the dataset into train-val-test.'
    )
    parser.add_argument('-p','--path_prefix',
                        help='Relative path to the dataset.',
                        default='./data/Kather_texture_2016_image_tiles_5000/'
                            + 'Kather_texture_2016_image_tiles_5000',
                        type=str)
    parser.add_argument('-sp','--split_path_prefix',
                        help='Path to store data after splitting.',
                        default='./data-split',
                        type=str)
    parser.add_argument('-t','--train_split',
                        help='Percentage of train set.',
                        default=0.85,
                        type=float)
    parser.add_argument('-v','--val_split',
                        help='Percentage of validation set.',
                        default=0.07,
                        type=float)
    args = parser.parse_args()

    class_names = os.listdir(args.path_prefix)

    for i, class_name in enumerate(class_names):
        split_class(
            class_name,
            args.path_prefix,
            args.split_path_prefix,
            args.train_split,
            args.val_split
        )