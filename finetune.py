import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torch

from video_transformer import ViViT
from transformer import ClassificationHead

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrain_pth='./vivit_model.pth'
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
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.005) #temporary for now
        return optimizer

#def load_Dataloader():


if __name__ =="__main__":
    # initialize ViVit from Video_Transformer file
    model = ViViT(pretrain_pth=pretrain_pth, weights_from='kinetics',
                  img_size=224,
                  num_frames=16,
                  attention_type='fact_encoder')
    # initialize Classhead and load in pretrian file
    cls_head = ClassificationHead(num_classes=num_class, in_channels=768)

    finetuner=FineTuneViVit(model, cls_head)

    print(finetuner)


