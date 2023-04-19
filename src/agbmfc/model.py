import logging
from typing import Optional, Literal
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.agbmfc.loading import chip_tensor_to_pixel_tensor


TrainReport = namedtuple('TrainReport',
                         ['train_loss_log', 'val_loss_log', 'last_epoch_loss_val', 'last_epoch_loss_train'])


class TrivialPixelRegressor(torch.nn.Module):
    def __init__(self, prediction=0):
        super().__init__()

        self.prediction = prediction

    def forward(self, ddict):
        batch_len = len(ddict["bands"])
        logits = torch.torch.zeros((batch_len, 1)) + self.prediction

        return logits

    @staticmethod
    def loss_fn(prediction, target):
        # print('x.shape', x.shape)
        loss = torch.sqrt(F.mse_loss(prediction, target, reduction="none") + 1e-8).mean()

        return loss


class PixelBLRegressor(TrivialPixelRegressor):
    def __init__(self, dim=256, seq_len=12, channels=4):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len
        self.channels = channels

        self.s1_band = torch.nn.Linear(channels, self.dim)
        self.s1_pos_embedding = torch.nn.Embedding(self.seq_len, self.dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.dim, nhead=8, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.regressor = torch.nn.Linear(self.dim, 1)

    def forward(self, ddict):
        _device = next(self.parameters()).device

        features = self.s1_band(ddict["bands"] / 100)

        bs = len(ddict["bands"])
        orders = torch.arange(12).repeat(bs).reshape((bs, self.seq_len)).to(_device)
        pos_emb = self.s1_pos_embedding(orders)

        assert features.shape == pos_emb.shape
        out = self.encoder(features + pos_emb)
        logits = self.regressor(out).squeeze(2)

        return logits[:, 0]


class PixelCls(torch.nn.Module):
    def __init__(self, dim=256, seq_len=12, channels=4, n_class=256):
        super().__init__()

        # super().__init__(dim=dim, seq_len=seq_len, channels=channels)

        self.dim = dim
        self.seq_len = seq_len
        self.channels = channels

        self.s1_band = torch.nn.Linear(channels, self.dim)
        self.s1_pos_embedding = torch.nn.Embedding(self.seq_len, self.dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.dim, nhead=8, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.n_class = n_class

        self.fc_inner = torch.nn.Linear(dim * seq_len, dim * seq_len // 4)
        self.fc_outer = torch.nn.Linear(dim * seq_len // 4, self.n_class)

    def forward(self, ddict):
        _device = next(self.parameters()).device

        features = self.s1_band(ddict["bands"] / 100)

        bs = len(ddict["bands"])
        orders = torch.arange(12).repeat(bs).reshape((bs, self.seq_len)).to(_device)
        pos_emb = self.s1_pos_embedding(orders)

        assert features.shape == pos_emb.shape
        out = self.encoder(features + pos_emb)
        # print(out.shape, flush=True)

        out = torch.flatten(out, 1)
        out = F.relu(self.fc_inner(out))
        logits = self.fc_outer(out)

        return logits

    @staticmethod
    def loss_fn(prediction, target):
        loss = torch.nn.CrossEntropyLoss()
        target = target.round().clip(min=0, max=255)
        target = target.type(torch.int64)

        # print(prediction.device, prediction.dtype)
        # print(target.device, target.dtype, target.max())

        return loss(prediction, target)


class PixelBLRegressorTinyInits(PixelBLRegressor):
    def __init__(self, dim=256, seq_len=12, channels=4):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len
        self.channels = channels

        self.s1_band = torch.nn.Linear(channels, self.dim)
        self.s1_pos_embedding = torch.nn.Embedding(self.seq_len, self.dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.dim, nhead=8, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.regressor = torch.nn.Linear(self.dim, 1)
        self.regressor.weight.data.fill_(0.01)
        self.regressor.bias.data.fill_(0.01)


class PixelMagnitudeSignalRegressor(PixelBLRegressor):
    def __init__(self, dim=256, seq_len=12, channels=4):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len
        self.channels = channels

        self.s1_band = torch.nn.Linear(channels, self.dim)
        self.s1_pos_embedding = torch.nn.Embedding(self.seq_len, self.dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.dim, nhead=8, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.magnitude_regressor = torch.nn.Linear(self.dim, 1)
        self.magnitude_regressor.weight.data.fill_(0.01)
        self.magnitude_regressor.bias.data.fill_(0.01)

        self.signal_regressor = torch.nn.Linear(self.dim, 1)

    def forward(self, ddict):
        _device = next(self.parameters()).device

        features = self.s1_band(ddict["bands"] / 100)

        bs = len(ddict["bands"])
        orders = torch.arange(12).repeat(bs).reshape((bs, self.seq_len)).to(_device)
        pos_emb = self.s1_pos_embedding(orders)

        assert features.shape == pos_emb.shape
        out = self.encoder(features + pos_emb)
        logits = self.magnitude_regressor(out).squeeze(2) + self.signal_regressor(out).squeeze(2)

        return logits[:, 0]


class PixelMagnitudeSignalMultRegressor(TrivialPixelRegressor):
    def __init__(self, dim=256, seq_len=12, channels=4):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len
        self.channels = channels

        self.s2_band = torch.nn.Linear(channels, self.dim)
        self.pos_embedding = torch.nn.Embedding(self.seq_len, self.dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.dim, nhead=8, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.magnitude_regressor = torch.nn.Linear(self.dim, 1)
        self.magnitude_regressor.weight.data.fill_(0.0)
        self.magnitude_regressor.bias.data.fill_(0.0)

        self.signal_regressor = torch.nn.Linear(self.dim, 1)

    def forward(self, ddict):
        _device = next(self.parameters()).device

        features = self.s2_band(ddict["bands"] / 100)

        bs = len(ddict["bands"])
        orders = torch.arange(12).repeat(bs).reshape((bs, self.seq_len)).to(_device)
        pos_emb = self.pos_embedding(orders)

        assert features.shape == pos_emb.shape
        out = self.encoder(features + pos_emb)

        magnitude = self.magnitude_regressor(out)
        magnitude = torch.exp(magnitude)
        logits = self.magnitude_regressor(out).squeeze(2) + magnitude.squeeze(2)

        return logits[:, 0]


class PixelMagnitudeSignalMultRegressorHonest(PixelMagnitudeSignalMultRegressor):

    def forward(self, ddict):
        _device = next(self.parameters()).device

        features = self.s2_band(ddict["bands"] / 100)

        bs = len(ddict["bands"])
        orders = torch.arange(12).repeat(bs).reshape((bs, self.seq_len)).to(_device)
        pos_emb = self.pos_embedding(orders)

        assert features.shape == pos_emb.shape
        out = self.encoder(features + pos_emb)

        magnitude = self.magnitude_regressor(out)
        magnitude = torch.exp(magnitude)
        logits = self.magnitude_regressor(out).squeeze(2) * magnitude.squeeze(2)

        return logits[:, 0]


def evaluate(model, val_dataloader, device):
    model.eval()

    loss_buffer = []
    with torch.no_grad():
        for batch, target in tqdm(val_dataloader, desc=f'evaluation'):
            a, b, sq, ch = batch.shape
            batch = batch.reshape(-1, sq, ch)
            target = target.reshape(-1)
            batch, target = batch.to(device), target.to(device)

            ddict = {
                'bands': batch
            }
            prediction = model(ddict)

            loss = model.loss_fn(prediction, target).item()
            loss_buffer.append(loss)

    mean_loss = np.mean(loss_buffer)
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
        prediction = model(ddict)

        loss = model.loss_fn(prediction, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    return losses


def inference(model: torch.nn.Module, chip_tensor, batch_size: int = 10_000) -> torch.Tensor:
    if batch_size < 0:
        raise ValueError('Batch size must be zero or positive')

    pixel_tensor = chip_tensor_to_pixel_tensor(chip_tensor)
    model.eval()
    with torch.no_grad():
        if batch_size:
            ...
            preds = []
            for batch_tensor in torch.split(pixel_tensor, batch_size, ):
                ddict = {
                    'bands': batch_tensor,
                }
                preds.append(model(ddict))
            prediction = torch.cat(preds)

        else:
            ddict = {
                'bands': pixel_tensor,
            }
            prediction = model(ddict)

    return prediction.reshape(256, 256)


def train(model, train_dataloader, val_dataloader, optimizer, device="cuda:0", n_epochs=10, scheduler=None):
    # TODO: add saving model

    model.to(device)

    train_loss_log = []
    val_loss_log = {
        'idx': [],
        'value': [],
    }

    for epoch in tqdm(range(n_epochs), ):
        epoch_losses = train_one_epoch(model, train_dataloader, optimizer, epoch, device)
        #         print(f'{epoch} - epoch_losses', epoch_losses )
        scheduler.step()
        train_loss_log.extend(epoch_losses)

        val_loss = evaluate(model, val_dataloader, device)
        val_loss_log['idx'].append(len(train_loss_log) - 1)
        val_loss_log['value'].append(val_loss)

        torch.save(train_loss_log, 'train_loss_log.pt')
        torch.save(val_loss_log, 'val_loss_log.pt')

        report = TrainReport(train_loss_log=train_loss_log, val_loss_log=val_loss_log,
                             last_epoch_loss_train=np.mean(epoch_losses), last_epoch_loss_val=val_loss)

        yield report


def pickup_model(model_kind: Optional[Literal['trivial', 'baseline-pixel']] = 'trivial') -> torch.nn.Module:

    if model_kind == 'trivial':
        path_weights = r'models/trivial-model.pt'
        model = TrivialPixelRegressor()

    elif model_kind == 'baseline-pixel':
        path_weights = r'models/baseline-pixel-model.pt'
        model = PixelBLRegressor()

    else:
        raise ValueError(f'unknown model_kind passed: {model_kind}')

    #  TODO:   check if it exist and load or return empty model
    return model
