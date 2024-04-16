import numpy as np
from sklearn.decomposition import NMF
import cv2
import staintools
from PIL import Image

# image = cv2.imread('data-split/test/01_TUMOR/3F85_CRC-Prim-HE-02_013.tif_Row_1_Col_1.tif')
# hematoxylin, eosin, _ = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# def vahadane_normalizer(stains):
#     nmf = NMF(n_components=2, init='random', random_state=0)
#     W = nmf.fit_transform(stains)
#     H = nmf.components_
#     return np.dot(W, H)

# normalized_hematoxylin = vahadane_normalizer(hematoxylin)
# normalized_eosin = vahadane_normalizer(eosin)

# normalized_image = np.stack((normalized_hematoxylin, normalized_eosin, np.zeros_like(normalized_hematoxylin)), axis=-1)

# cv2.imwrite('normalized_image.jpg', normalized_hematoxylin)
# cv2.imshow('Normalized Image', normalized_hematoxylin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def vahadane_normalizer(ref_path, method, target_path):

import matplotlib.pyplot as plt

ref_path = './data-split/train/03_COMPLEX/1DA3_CRC-Prim-HE-06_028.tif_Row_1_Col_151.tif'

target_path = "./data-split/train/06_MUCOSA/1A62_CRC-Prim-HE-05.tif_Row_1201_Col_1501.tif"

ref_image = Image.open(ref_path)
target_image = Image.open(target_path)

ref_rgb = ref_image.convert("RGB")
tar_rgb = target_image.convert("RGB")

# Convert to a numpy array
ref_np = np.array(ref_rgb)
tar_np = np.array(tar_rgb)
# Assert type to be uint8
ref_np = ref_np.astype(np.uint8)
tar_np = tar_np.astype(np.uint8)

# pImg_ref=Image.fromarray(ref_image, mode='RGB')
# pImg_tar=Image.fromarray(target_image, mode='RGB')

normalizer = staintools.StainNormalizer(method='vahadane')

# Fit the normalizer on the reference image
normalizer.fit(ref_np)

# Transform the target image to match the reference
normalized_image = normalizer.transform(tar_np)

# Display the images
fig, axes = plt.subplots(1, 3, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(ref_np)
ax[0].set_title("Reference")
ax[0].axis('off')  # Remove axes ticks

ax[1].imshow(tar_np)
ax[1].set_title("Target")
ax[1].axis('off')  # Remove axes ticks

ax[2].imshow(normalized_image)
ax[2].set_title("Normalized Target")
ax[2].axis('off')  # Remove axes ticks
plt.show()