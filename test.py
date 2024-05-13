import os

# Torch Imports
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet

# Local Imports
from dataset import CRCTissueDataset
from utility import run_epoch

# Constants
RUN_ID = "5"
IMG_SIZE = (512, 512)
TEST_IMG_PATH = "./data-split/test"
LOG_PATH = os.path.join("./logs", RUN_ID)
checkpoint_file = "checkpoint39.pt"
CHECKPOINT_PATH = f"./checkpoints/{RUN_ID}/{checkpoint_file}"

# Hyperparameters
batch_size = 16
loss_fn = nn.CrossEntropyLoss()
normalizer = None

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = EfficientNet.from_name('efficientnet-b7', num_classes=8)
checkpoint = torch.load(CHECKPOINT_PATH)

# Load Test data
test_transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=IMG_SIZE),
            v2.ToDtype(torch.float, scale=True)
        ])

test_dataset = CRCTissueDataset(
    imgs_path=TEST_IMG_PATH,
    normalizer=normalizer,
    transforms=test_transforms
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

test_logger = SummaryWriter(
    log_dir=os.path.join(LOG_PATH, "test")
)

model.load_state_dict(checkpoint)
model = model.to(device)

# Testing
model.eval()

test_metrics = run_epoch(
    test_dataloader,
    model,
    device,
    loss_fn,
    test_logger)

print(
        f"Test: Loss - {test_metrics[0]}, " +
        f"Accuracy - {test_metrics[1]}, " +
        f"Specificity - {test_metrics[2]}, " +
        f"Precision - {test_metrics[3]}, " +
        f"Recall - {test_metrics[4]}, " +
        f"F1 - {test_metrics[5]}"
    )