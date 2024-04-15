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

ref_path = "./data-split/1A11_CRC-Prim-HE-07_022.tif_Row_601_Col_151.tif"
target_path = './data-split/train/07_ADIPOSE/10A0_CRC-Prim-HE-05_032.tif_Row_1051_Col_1501.tif'

ref_image = cv2.imread(ref_path)
target_image = cv2.imread(target_path)

pImg_ref=Image.fromarray(ref_image, mode='RGB')
pImg_tar=Image.fromarray(target_image, mode='RGB')

normalizer = staintools.StainNormalizer(method='vahadane')

# Fit the normalizer on the reference image
normalizer.fit(ref_image)

# Transform the target image to match the reference
normalized_image = normalizer.transform(target_image)