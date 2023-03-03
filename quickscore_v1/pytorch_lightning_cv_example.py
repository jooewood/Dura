#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
from copy import deepcopy
from pathlib import Path
from typing import Union, Optional, Callable

from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn, sigmoid, optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import ConcatDataset, Subset, DataLoader, Dataset
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger, LoggerCollection

DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"


class _WrappedDataset:
    """Allows to add transforms to a given Dataset."""
    def __init__(self,
                 dataset: Dataset,
                 transform: Optional[Callable] = None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample, label = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label


class CatsDogsDataCV:
    """Cats & dogs toy dataset for cross-validation."""
    def __init__(self,
                 data_dir: Union[str, Path],
                 num_workers: int = 16,
                 batch_size: int = 32,
                 n_splits: int = 5,
                 stratify: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

        # Cross-validation
        self.n_splits = n_splits
        self.stratify = stratify

        # Data normalization
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]

    def prepare_data(self):
        """Download the raw data."""
        download_and_extract_archive(url=DATA_URL,
                                     download_root=str(self.data_dir),
                                     remove_finished=True)

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=self._mean, std=self._std)

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def val_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def get_splits(self):
        if self.stratify:
            labels = self.get_data_labels()
            cv_ = StratifiedKFold(n_splits=self.n_splits)
        else:
            labels = None
            cv_ = KFold(n_splits=self.n_splits)

        dataset = self.get_dataset()
        n_samples = len(dataset)
        for train_idx, val_idx in cv_.split(X=range(n_samples), y=labels):
            _train = Subset(dataset, train_idx)
            self._update_mean_std(dataset=_train)
            train_dataset = _WrappedDataset(_train, self.train_transform)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers)

            _val = Subset(dataset, val_idx)
            val_dataset = _WrappedDataset(_val, self.val_transform)
            val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers)

            yield train_loader, val_loader

    def _update_mean_std(self, dataset):
        """Computes the mean and std of the given (image) dataset.

        Instantiates a dataloader to compute the mean and std from batches.
        """
        _dataset = _WrappedDataset(dataset=dataset,
                                   transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                 transforms.ToTensor()]))
        _dataloader = DataLoader(dataset=_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)
        mean, std, n_samples = 0., 0., 0.
        for images, _ in _dataloader:
            batch_samples = images.size(0)
            data = images.view(batch_samples, images.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            n_samples += batch_samples
        self._mean = mean / n_samples
        self._std = std / n_samples

    def get_dataset(self):
        """Creates and returns the complete dataset."""
        train_data_path = Path(self.data_dir).joinpath('cats_and_dogs_filtered', 'train')
        train_dataset = ImageFolder(root=train_data_path)
        valid_data_path = Path(self.data_dir).joinpath('cats_and_dogs_filtered', 'validation')
        valid_dataset = ImageFolder(root=valid_data_path)
        return ConcatDataset([train_dataset, valid_dataset])

    def get_data_labels(self):
        dataset = self.get_dataset()
        return [int(sample[1]) for sample in dataset]


class CV:
    """Cross-validation with a LightningModule."""
    def __init__(self,
                 *trainer_args,
                 **trainer_kwargs):
        super().__init__()
        self.trainer_args = trainer_args
        self.trainer_kwargs = trainer_kwargs

    @staticmethod
    def _update_logger(logger, fold_idx: int):
        if hasattr(logger, 'experiment_name'):
            logger_key = 'experiment_name'
        elif hasattr(logger, 'name'):
            logger_key = 'name'
        else:
            raise AttributeError('The logger associated with the trainer '
                                 'should have an `experiment_name` or `name` '
                                 'attribute.')
        new_experiment_name = getattr(logger, logger_key) + f'/{fold_idx}'
        setattr(logger, logger_key, new_experiment_name)

    @staticmethod
    def update_modelcheckpoint(model_ckpt_callback, fold_idx):
        _default_filename = '{epoch}-{step}'
        _suffix = f'_fold{fold_idx}'
        if model_ckpt_callback.filename is None:
            new_filename = _default_filename + _suffix
        else:
            new_filename = model_ckpt_callback.filename + _suffix
        setattr(model_ckpt_callback, 'filename', new_filename)

    def update_logger(self, trainer: Trainer, fold_idx: int):
        if not isinstance(trainer.logger, LoggerCollection):
            _loggers = [trainer.logger]
        else:
            _loggers = trainer.logger

        # Update loggers:
        for _logger in _loggers:
            self._update_logger(_logger, fold_idx)

    def fit(self, model: LightningModule, data: CatsDogsDataCV):
        splits = data.get_splits()
        for fold_idx, loaders in enumerate(splits):

            # Clone model & instantiate a new trainer:
            _model = deepcopy(model)
            trainer = Trainer(*self.trainer_args, **self.trainer_kwargs)

            # Update loggers and callbacks:
            self.update_logger(trainer, fold_idx)
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    self.update_modelcheckpoint(callback, fold_idx)

            # Fit:
            trainer.fit(_model, *loaders)


class MyCustomModel(LightningModule):
    """Custom classification model."""

    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr

        self.__build_model()

    def __build_model(self):
        # Classifier:
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(3 * 3 * 64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

        # Loss:
        self.loss = binary_cross_entropy_with_logits

        # Metrics:
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)

        # 2. Compute loss
        train_loss = self.loss(y_logits, y_true)

        # 3. Compute accuracy:
        train_accuracy = self.train_acc(sigmoid(y_logits), y_true.int())
        self.log("train_acc", train_accuracy, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)

        # 2. Compute loss
        self.log("val_loss", self.loss(y_logits, y_true), prog_bar=True)

        # 3. Compute accuracy:
        valid_accuracy = self.valid_acc(sigmoid(y_logits), y_true.int())
        self.log("val_acc", valid_accuracy, prog_bar=True)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        return optimizer


if __name__ == '__main__':

    # Trainer
    neptune_logger = NeptuneLogger(project_name=NEPTUNE_PROJECT_NAME,
                                   experiment_name=NEPTUNE_EXPERIMENT_NAME)

    model_checkpoint = ModelCheckpoint(dirpath=MODEL_CHECKPOINT_DIR_PATH,
                                       monitor='val_acc',
                                       save_top_k=1,
                                       mode='max',
                                       filename='custom_model_{epoch}',)

    trainer_kwargs_ = {'weights_summary': None,
                       'progress_bar_refresh_rate': 1,
                       'num_sanity_val_steps': 0,
                       'gpus': [0],
                       'max_epochs': 10,
                       'logger': neptune_logger,
                       'callbacks': [model_checkpoint]}

    cv = CV(**trainer_kwargs_)

    # LightningModule
    clf = MyCustomModel(lr=1e-3)

    # Run a 5-fold cross-validation experiment:
    image_data = CatsDogsDataCV(data_dir=DATA_DIR, n_splits=5, stratify=False)

    cv.fit(clf, image_data)

