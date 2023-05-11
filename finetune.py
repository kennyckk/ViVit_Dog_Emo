import os
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
import torch

from video_transformer import ViViT
from transformer import ClassificationHead
from data_trainer import DogDataModule

pretrain_pth='/content/drive/MyDrive/ViViT_Weight/vivit_model.pth' #for google collab testing
num_class=2

class FineTuneViVit(pl.LightningModule):
    def __init__(self, model, cls_head):
        super().__init__()
        self.model = model
        self.cls_head = cls_head

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        inputs, labels = batch # input already are (T C H W)

        preds= self.model(inputs)
        preds = self.cls_head(preds)
        loss = nn.CrossEntropyLoss()(preds, labels)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        inputs, labels = batch # input already are (T C H W)

        preds = self.model(inputs)
        preds = self.cls_head(preds)
        val_loss = nn.CrossEntropyLoss()(preds, labels)

        self.log("val_loss" , val_loss,prog_bar=True)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.005) #temporary for now
        return optimizer




if __name__ =="__main__":
    # initialize ViVit from Video_Transformer file
    model = ViViT(pretrain_pth=pretrain_pth, weights_from='kinetics',
                  img_size=224,
                  num_frames=16,
                  attention_type='fact_encoder')
    # initialize Classhead and load in pretrian file
    cls_head = ClassificationHead(num_classes=num_class, in_channels=768)

    # initiate Model Trainer
    finetuner=FineTuneViVit(model, cls_head)
    #print(finetuner)

    # intialize DataModule
    dm= DogDataModule(batch_size=2,
                      num_workers=0,
                      train_ann_path='/content/drive/MyDrive/dogdata/train.csv',
                      val_ann_path='/content/drive/MyDrive/dogdata/eval.csv'
                      )

    trainer=Trainer(devices='auto',
                    accelerator='auto',
                    max_epochs=20,
                    log_every_n_steps=10,
                    check_val_every_n_epoch=1)

    trainer.fit(finetuner,dm)

