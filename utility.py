from tqdm import tqdm
import torch
from metrics import get_metrics
import numpy as np

def normalize_image(ref_img, tar_img, normalizer):
    # Convert image to RGB uint8
    tar_img = tar_img.convert("RGB")
    ref_img = ref_img.convert("RGB")
    tar_img = np.array(tar_img)
    ref_img = np.array(ref_img)
    tar_img = tar_img.astype(np.uint8)
    ref_img = ref_img.astype(np.uint8)
    # Fit the normalizer on reference image
    normalizer.fit(ref_img)
    # Normalize the target image
    tar_img = normalizer.transform(tar_img)

    return tar_img

def run_epoch(dataloader,
              model,
              device,
              loss_fn,
              logger,
              opt=None,
              n_classes=8,
              step=0):

    loss_list = []
    acc_list = []
    spec_list = []
    prec_list = []
    rec_list = []
    f1_list = []

    for x, y in tqdm(dataloader):

        # Moving input to device
        x = x.to(device)
        y = y.to(device)

        # Running forward propagation
        y_hat = model(x)

        loss = loss_fn(y_hat, y)

        if opt is not None:
            # Make all gradients zero.
            opt.zero_grad()

            # Run backpropagation
            loss.backward()

            # Clipping gradients to 0.01
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

            # Update parameters
            opt.step()

        loss_list.append(loss.item())

        # detach removes y_hat from the original computational graph which might be on gpu.
        y_hat = y_hat.detach().cpu()
        y = y.cpu()

        # Compute metrics
        acc = get_metrics(y_hat, y, metric="accuracy")
        acc_list.append(acc)

        spec = get_metrics(y_hat, y, metric="specificity")
        spec_list.append(spec)

        prec = get_metrics(y_hat, y, metric="precision")
        prec_list.append(prec)

        rec = get_metrics(y_hat, y, metric="recall")
        rec_list.append(rec)

        f1 = get_metrics(y_hat, y, metric="f1")
        f1_list.append(f1)

        logger.add_scalar(f"loss", loss.item(), step)
        logger.add_scalar(f"accuracy", acc, step)
        logger.add_scalar(f"specificity", spec.mean(), step)
        logger.add_scalar(f"precision", prec.mean(), step)
        logger.add_scalar(f"recall", rec.mean(), step)
        logger.add_scalar(f"f1", f1.mean(), step)

        for j in range(n_classes):
            logger.add_scalar(f"precision/{j}", prec[j], step)
            logger.add_scalar(f"recall/{j}", rec[j], step)
            logger.add_scalar(f"f1/{j}", f1[j], step)
            logger.add_scalar(f"specificity/{j}", spec[j], step)

        step += 1

    avg_loss = torch.Tensor(loss_list).mean()
    avg_acc = torch.Tensor(acc_list).mean()
    avg_spec = torch.vstack(spec_list).mean()
    avg_p = torch.vstack(prec_list).mean()
    avg_r = torch.vstack(rec_list).mean()
    avg_f1 = torch.vstack(f1_list).mean()

    return avg_loss, avg_acc, avg_spec, avg_p, avg_r, avg_f1