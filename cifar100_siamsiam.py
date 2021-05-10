from lightly.utils.benchmarking import knn_predict
from mlp_eval import SSLEvaluator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
import lightly
from lightly.utils import BenchmarkModule

import torch.distributed as dist
from torchvision.datasets import CIFAR100

cifar_normalize = {
    'mean': [0.5071, 0.4866, 0.4409],
    'std': [0.2009, 0.1984, 0.2023]
}


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


num_workers = 8
# set max_epochs to 800 for long run (takes around 10h on a single V100)
MAX_EPOCHS = 800
LM_EVAL_TRAIN_EPOCHS = 150
LM_EVAL_EPOCHS = 80
knn_k = 200
knn_t = 0.1
classes = 100

# use a GPU if available
#gpus = -1 if torch.cuda.is_available() else 0
gpus = [3]
distributed_backend = 'ddp' if len(gpus) > 1 else None

# Use SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(input_size=32,
                                                gaussian_blur=0.,
                                                normalize=cifar_normalize)

# No additional augmentations for the test set
to_tensor = torchvision.transforms.ToTensor()
to_pil = torchvision.transforms.ToPILImage()
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=cifar_normalize['mean'],
        std=cifar_normalize['std'],
    )
])

DATASET_DIR = "/data/datasets/cifar100"
cifar100_train = CIFAR100(DATASET_DIR, train=True)
cifar100_train_knn = CIFAR100(DATASET_DIR,
                              train=True,
                              transform=test_transforms)
cifar100_test = CIFAR100(DATASET_DIR, train=False, transform=test_transforms)
dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(
    cifar100_train)
# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(
    cifar100_train_knn)
dataset_test = lightly.data.LightlyDataset.from_torch_dataset(cifar100_test)

# print('ttrain_ssl:')
# print(dataset_train_ssl[0])
# print('ttrain_knn:')
# print(dataset_train_kNN[0])
# print('test:')
# print(dataset_test[0])


def get_data_loaders(batch_size: int):
    dataloader_train_ssl = torch.utils.data.DataLoader(dataset_train_ssl,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       collate_fn=collate_fn,
                                                       drop_last=True,
                                                       num_workers=num_workers)

    dataloader_train_kNN = torch.utils.data.DataLoader(dataset_train_kNN,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       drop_last=False,
                                                       num_workers=num_workers)

    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=num_workers)

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test


class SimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        self.resnet_simsiam = \
            lightly.models.SimSiam(self.backbone, num_ftrs=512, num_mlp_layers=2)
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
        self.num_classes = num_classes
        self.train_epochs = 0

    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simsiam(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(),
                                lr=6e-2,
                                momentum=0.9,
                                weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, MAX_EPOCHS)
        return [optim], [scheduler]

    def _is_eval_needed(self):
        if (self.train_epochs == 1
                or ((self.train_epochs % LM_EVAL_EPOCHS) == 0)
                or (self.train_epochs == MAX_EPOCHS)):
            return True
        else:
            return False

    def training_epoch_end(self, outputs):
        self.train_epochs += 1

        if not self._is_eval_needed():
            return

        # train new linear head
        self.backbone.eval()

        print('Updating features bank')

        # update feature bank after training
        self.backbone.eval()
        self.feature_bank = []
        self.targets_bank = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                img = img.to(self.dummy_param.device)
                target = target.to(self.dummy_param.device)
                feature = self.backbone(img).squeeze()
                # feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(target)
        self.feature_bank = torch.cat(self.feature_bank,
                                      dim=0)  #.t().contiguous()
        self.targets_bank = torch.cat(self.targets_bank,
                                      dim=0)  #.t().contiguous()

        print('Training linear classifier...')

        print('Features bank shape:', self.feature_bank.shape)

        eval_nepochs = LM_EVAL_TRAIN_EPOCHS
        head_classifier_lr = 5e-3
        head_classifier_min_lr = 1e-6
        head_classifier_lr_patience = 3
        head_classifier_hidden_mlp = 0

        self._task_classifier = SSLEvaluator(self.feature_bank.shape[1],
                                             self.num_classes,
                                             head_classifier_hidden_mlp, 0.0)
        self._task_classifier.to(self.dummy_param.device)
        _task_classifier_optimizer = torch.optim.Adam(
            self._task_classifier.parameters(), lr=head_classifier_lr)

        # train on train dataset after learning representation of task
        classifier_train_step = 0
        self._task_classifier.train()

        lm_train_ds = torch.utils.data.TensorDataset(self.feature_bank,
                                                     self.targets_bank)
        lm_train_dl = torch.utils.data.DataLoader(lm_train_ds,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0)

        for e in range(eval_nepochs):
            # train
            train_loss = 0.0
            train_samples = 0.0
            for _x, y in lm_train_dl:
                # _x = self.backbone(img_1.to(self.dummy_param.device))
                # _x = _x.reshape(_x.size(0), -1)
                # _x = _x.detach()  # make sure we don't backprop through encoder
                # forward pass
                mlp_preds = self._task_classifier(_x)
                mlp_loss = F.cross_entropy(mlp_preds, y)
                # update finetune weights
                mlp_loss.backward()
                _task_classifier_optimizer.step()
                _task_classifier_optimizer.zero_grad()
                train_loss += mlp_loss.item()
                train_samples += len(y)

                # val_acc = self._accuracy(mlp_preds, y)
                # pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
                # self.logger.tbwriter.add_scalar(f"t{t}/{name}-loss", mlp_loss, classifier_train_step)
                # self.logger.log_scalar(task=t, iter=classifier_train_step, name=f"{name}-loss", value=mlp_loss.item(), group="train")
                classifier_train_step += 1

            train_loss = train_loss / train_samples
            print(f'Epoch: {e} Loss: {train_loss}')

        # normalize features for kNN
        self.feature_bank = F.normalize(self.feature_bank.squeeze(), dim=1).t().contiguous()
        self.targets_bank = self.targets_bank.t().contiguous()
        self._task_classifier.eval()
        self.backbone.train()


    def validation_step(self, batch, batch_idx):
        if self._is_eval_needed():
            # we can only do lin. class predictions once we have it
            if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank') and hasattr(self, '_task_classifier'):
                images, targets, _ = batch
                feature = self.backbone(images).squeeze()

                # kNN eval
                feature_norm = F.normalize(feature, dim=1)
                pred_labels = knn_predict(feature_norm, self.feature_bank,
                                          self.targets_bank, self.num_classes,
                                          self.knn_k, self.knn_t)
                num = images.size()
                top1 = (pred_labels[:, 0] == targets).float().sum()
                lm_preds = self._task_classifier(feature)
                lm_corr = (lm_preds.argmax(1) == targets).sum().cpu().item()
                results = (num, top1, lm_corr)
                print('Validation batch: ', batch_idx, ' results:', results)
                return results

    def validation_epoch_end(self, outputs):
        if outputs and self._is_eval_needed():
            device = self.dummy_param.device
            total_num = torch.Tensor([0]).to(device)
            total_top1 = torch.Tensor([0.]).to(device)
            total_lm_corr = torch.Tensor([0.]).to(device)
            for (num, top1, lm_corr) in outputs:
                total_num += num[0]
                total_top1 += top1
                total_lm_corr += lm_corr

            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(total_num)
                dist.all_reduce(total_top1)
                dist.all_reduce(total_lm_corr)

            acc_knn = float(total_top1.item() / total_num.item())
            acc_lm = float(total_lm_corr.item() / total_num.item())
            # if acc > self.max_accuracy:
            # self.max_accuracy = acc
            self.log('kNN_accuracy', acc_knn * 100.0, prog_bar=True)
            self.log('LM_accuracy', acc_lm * 100.0, prog_bar=True)


# benchmark
n_runs = 1  # optional, increase to create multiple runs and report mean + std
SEEDS = [1993]
batch_sizes = [512]
model_names = ['SimSiam_512']
models = [SimSiamModel]

# results
bench_results = []
gpu_memory_usage = []

# loop through configurations and train models
for batch_size in batch_sizes:
    for BenchmarkModel in models:
        runs = []
        #for seed in range(n_runs):
        for seed in [SEEDS]:
            pl.seed_everything(seed)
            dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
                batch_size)
            benchmark_model = BenchmarkModel(dataloader_train_kNN, classes)
            trainer = pl.Trainer(
                max_epochs=MAX_EPOCHS,
                gpus=gpus,
                progress_bar_refresh_rate=20,
                distributed_backend=distributed_backend,
                check_val_every_n_epoch=LM_EVAL_EPOCHS,
            )
            trainer.fit(benchmark_model,
                        train_dataloader=dataloader_train_ssl,
                        val_dataloaders=dataloader_test)
            gpu_memory_usage.append(torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            runs.append(benchmark_model.max_accuracy)

            # delete model and trainer + free up cuda memory
            del benchmark_model
            del trainer
            torch.cuda.empty_cache()
        bench_results.append(runs)

for result, model, gpu_usage in zip(bench_results, model_names,
                                    gpu_memory_usage):
    result_np = np.array(result)
    mean = result_np.mean()
    std = result_np.std()
    print(
        f'{model}: {mean:.2f} +- {std:.2f}, GPU used: {gpu_usage / (1024.0**3):.1f} GByte',
        flush=True)
