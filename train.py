import os

# Torch Imports
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

# Local Imports
from dataset import CRCTissueDataset
# from model import CRCTissueClassifier
from utility import run_epoch

# Constants
RUN_ID = "5"
TRAIN_IMG_PATH = "./data-split/train"
VAL_IMG_PATH = "./data-split/val"
CKPT_PATH = os.path.join("./checkpoints/", RUN_ID)
LOG_PATH = os.path.join("./logs", RUN_ID)
IMG_SIZE = (512, 512)
N_CLASSES = 8

# Hyperparams
lr = 1e-4
lmbda = 1e-5
batch_size = 16
n_epochs = 40

# Normalizer
# normalizer = staintools.StainNormalizer(method='vahadane')

# Trainset augmentation
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=IMG_SIZE),
    v2.RandomRotation(degrees=20),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.ToDtype(torch.float, scale=True),
])

# Load train data
train_dataset = CRCTissueDataset(
    imgs_path=TRAIN_IMG_PATH,
    transforms=train_transforms)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

# Validation set augmentation
val_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=IMG_SIZE),
    v2.ToDtype(torch.float, scale=True),
])
# Load validation data
val_dataset = CRCTissueDataset(
    imgs_path=VAL_IMG_PATH,
    transforms=val_transforms)
val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)

# Other hyperparameters
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = CRCTissueClassifier()
model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=8)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=lr, weight_decay=lmbda)

# Logging
train_logger = SummaryWriter(log_dir=os.path.join(LOG_PATH, "train"))
val_logger = SummaryWriter(log_dir=os.path.join(LOG_PATH, "val"))

model = model.to(device)

global_step = 0

os.makedirs(CKPT_PATH, exist_ok=True)

# Logging model graph
x, _ = next(iter(train_dataloader))
train_logger.add_graph(model, x.to(device))

# Training
for i in range(n_epochs):
    model.train()
    # run one epoch of training
    train_metrics = run_epoch(
        train_dataloader,
        model,
        device,
        loss_fn,
        train_logger,
        opt=opt,
        step=i*len(train_dataloader)
    )

    model.eval()
    # run one epoch of validation
    val_metrics = run_epoch(
        val_dataloader,
        model,
        device,
        loss_fn,
        val_logger,
        step=i*len(val_dataloader)
    )

    print(f"Epoch {i}:")
    print(
        f"Train: Loss - {train_metrics[0]}, " +
        f"Accuracy - {train_metrics[1]}, " +
        f"Specificity - {train_metrics[2]}, " +
        f"Precision - {train_metrics[3]}, " +
        f"Recall - {train_metrics[4]}, " +
        f"F1 - {train_metrics[5]}"
    )

    print(
        f"Val: Loss - {val_metrics[0]}, " +
        f"Accuracy - {val_metrics[1]}, " +
        f"Specificity - {val_metrics[2]}, " +
        f"Precision - {val_metrics[3]}, " +
        f"Recall - {val_metrics[4]}, " +
        f"F1 - {val_metrics[5]}"
    )

    # Logging per epoch so that traning and val can be compared properly
    # Because of equal no. of data points on the graph
    train_logger.add_scalar(f"epoch/loss", train_metrics[0], i)
    train_logger.add_scalar(f"epoch/accuracy", train_metrics[1], i)
    train_logger.add_scalar(f"epoch/specificity", train_metrics[2], i)
    train_logger.add_scalar(f"epoch/precision", train_metrics[3], i)
    train_logger.add_scalar(f"epoch/recall", train_metrics[4], i)
    train_logger.add_scalar(f"epoch/f1", train_metrics[5], i)

    val_logger.add_scalar(f"epoch/loss", val_metrics[0], i)
    val_logger.add_scalar(f"epoch/accuracy", val_metrics[1], i)
    val_logger.add_scalar(f"epoch/specificity", val_metrics[2], i)
    val_logger.add_scalar(f"epoch/precision", val_metrics[3], i)
    val_logger.add_scalar(f"epoch/recall", val_metrics[4], i)
    val_logger.add_scalar(f"epoch/f1", val_metrics[5], i)

    torch.save(model.state_dict(), os.path.join(CKPT_PATH, f"checkpoint{i}.pt"))