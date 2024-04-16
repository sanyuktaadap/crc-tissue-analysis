import glob
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.decomposition import NMF
from utility import normalize_image

class CRCTissueDataset(Dataset):
    def __init__(
        self,
        imgs_path,
        normalizer,
        norm_reference_path,
        transforms=None,
        class_map={
            "tumor" : 0,
            "stroma" : 1,
            "complex" : 2,
            "lympho" : 3,
            "debris" : 4,
            "mucosa" : 5,
            "adipose" : 6,
            "empty" : 7
        }
    ):
        """Initializes a Colorectal Histology dataset instance
        Args:
            imgs_path (str): Path to the folder containing classwise images
            transforms (torchvision.transforms): Transforms to be run on an image
            class_map (dict, optional): Provides the mapping from the name to the number
        """
        # Path to dataset
        self.imgs_path = imgs_path
        # Classes dictionary
        self.class_map = class_map
        # Stain color normalization method
        self.normalizer = normalizer
        # Path to reference image for normalization
        self.norm_reference_path = norm_reference_path
        self.transforms = transforms
        # List containing paths to all the images
        self.data = []
        # List of all folders inside imgs_path
        file_list = glob.glob(os.path.join(self.imgs_path, "*"))
        # Iterate over all the classes in file list
        for class_path in file_list:
            # For each class, extract the actual class name
            class_name = class_path.split('_')[-1].lower()
            # Retrieve each image (.jpg) in class folders
            for img_path in glob.glob(os.path.join(class_path, "*")):
                # self.data will contain path to each image, the respective class
                self.data.append((img_path, class_name))

    def __len__(self):
        """Gets the length of the dataset
        Returns:
            int: total number of data points
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Gets the indexed items from the dataset
        Args:
            idx (int): index number
        Returns:
            vector, int: indexed image with its corresponding label
        """
        # idx - indexing data for accessibility
        img_path, class_name = self.data[idx]
        # Assigning ids to each class (number, not name of the class)
        class_id = self.class_map[class_name]
        # Loads an image from the given image_path
        img = Image.open(img_path)

        if self.normalizer is not None:
            ref_img = Image.open(self.norm_reference_path)
            img = normalize_image(ref_img, img, self.normalizer)

        if self.transforms:
            img = self.transforms(img)

        return img, class_id