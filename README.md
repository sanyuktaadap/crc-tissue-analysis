# Identify Tissue Textures in CRC

Using Colorectal Cancer Histology images to estimate tumor-stroma ratio by identifying the 8 types of tissue textures fund in CRC.

- To set up data for ML, split data using [run_data_setup](./run_data_setup.py) script

#### Train-Val-Test Scripts
- [Training and Validation](./train.py)
- [Testing](./test.py)


(a) tumour epithelium,
(b) simple stroma,
(c) complex stroma
(d) immune cells,
(e) debris and mucus,
(f) mucosal glands,
(g) adipose tissue,
(h) background

Reference papers:
1. https://www.nature.com/articles/s41598-019-50587-1
2.