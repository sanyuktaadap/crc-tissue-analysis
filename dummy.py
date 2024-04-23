import numpy as np
from sklearn.decomposition import NMF
import staintools
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import torch

img_path = './data-split/train/03_COMPLEX/1DA3_CRC-Prim-HE-06_028.tif_Row_1_Col_151.tif'
img = Image.open(img_path)

transforms = v2.Compose([
    v2.ToImage(),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.ToDtype(torch.float, scale=True),
])

out = transforms(img)

print(out)

# target_path = './data-split/train/03_COMPLEX/1DA3_CRC-Prim-HE-06_028.tif_Row_1_Col_151.tif'

# ref_path = "./data-split/test/08_EMPTY/10A7_CRC-Prim-HE-06_005.tif_Row_2251_Col_6451.tif"

# ref_image = Image.open(ref_path)
# target_image = Image.open(target_path)

# ref_rgb = ref_image.convert("RGB")
# tar_rgb = target_image.convert("RGB")

# # Convert to a numpy array
# ref_np = np.array(ref_rgb)
# tar_np = np.array(tar_rgb)
# # Assert type to be uint8
# ref_np = ref_np.astype(np.uint8)
# tar_np = tar_np.astype(np.uint8)

# # pImg_ref=Image.fromarray(ref_image, mode='RGB')
# # pImg_tar=Image.fromarray(target_image, mode='RGB')

# normalizer = staintools.StainNormalizer(method='vahadane')

# # Fit the normalizer on the reference image
# normalizer.fit(ref_np)

# # Transform the target image to match the reference
# normalized_image = normalizer.transform(tar_np)

# # Display the images
# fig, axes = plt.subplots(1, 3, figsize=(20, 20))
# ax = axes.ravel()

# ax[0].imshow(ref_np)
# ax[0].set_title("Reference")
# ax[0].axis('off')  # Remove axes ticks

# ax[1].imshow(tar_np)
# ax[1].set_title("Target")
# ax[1].axis('off')  # Remove axes ticks

# ax[2].imshow(normalized_image)
# ax[2].set_title("Normalized Target")
# ax[2].axis('off')  # Remove axes ticks
# plt.show()