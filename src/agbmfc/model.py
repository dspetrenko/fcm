from typing import Optional, Literal

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.agbmfc.loading import chip_tensor_to_pixel_tensor


class TrivialPixelRegressor(torch.nn.Module):
    def __init__(self, prediction=0):
        super().__init__()

        self.prediction = prediction

    def forward(self, ddict):
        batch_len = len(ddict["bands"])
        logits = torch.torch.zeros((batch_len, 1)) + self.prediction

        return logits

    def loss_fn(self, x, target):
        x_ = x[:, 0]
        y_ = target
        loss = torch.sqrt(F.mse_loss(x_, y_, reduction="none") + 1e-8).mean()
        return loss


def evaluate(model, val_dataloader, device):
    model.eval()

    loss_buffer = torch.empty(0)
    with torch.no_grad():
        for batch, target in tqdm(val_dataloader, desc=f'evaluation'):
            batch = batch[0]
            target = target[0]
            batch, target = batch.to(device), target.to(device)

            ddict = {
                'bands': batch
            }
            prediction = model(ddict)[:, 0]

            loss = model.loss_fn(target, prediction)
            loss_buffer = torch.cat((loss_buffer.cpu(), loss.cpu().unsqueeze(0)))

    mean_loss = np.mean(loss_buffer.cpu().detach().numpy())
    return mean_loss


def train_one_epoch(model, train_dataloader, optimizer, epoch, device="cuda:0", ) -> list:
    model.train()

    losses = []
    for batch, target in tqdm(train_dataloader, desc=f'training batch: epoch - {epoch}'):

        a, b, sq, ch = batch.shape
        batch = batch.reshape(-1, sq, ch)
        target = target.reshape(-1)

        batch, target = batch.to(device), target.to(device)
        #         all_targets = torch.cat((all_targets.cpu(), target.cpu()))
        ddict = dict()
        ddict["bands"] = batch
        prediction = model(ddict)[:, 0]

        loss = model.loss_fn(target, prediction)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    return losses


def inference(model: torch.nn.Module, chip_tensor) -> torch.Tensor:
    pixel_tensor = chip_tensor_to_pixel_tensor(chip_tensor)
    model.eval()
    with torch.no_grad():
        ddict = {
            'bands': pixel_tensor,
        }
        prediction = model(ddict)[:, 0].reshape(256, 256)

    return prediction


def pickup_model(model_kind: Optional[Literal['trivial']] = 'trivial') -> torch.nn.Module:

    if model_kind == 'trivial':
        path_weights = r'models/trivial-model.pt'
        model = TrivialPixelRegressor()

    else:
        raise ValueError(f'unknown model_kind passed: {model_kind}')

    model = torch.load(path_weights)
    return model
