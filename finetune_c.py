import os
import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from torchmetrics import Accuracy
from tqdm.auto import tqdm

from video_transformer import ViViT
from transformer import ClassificationHead
from data_transform import create_video_transform, TemporalRandomCrop,transforms_train_dog,transforms_eval
from dataset import DogDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Function to load in model
def load_model(pretrain_pth, num_class=2,drop_out=0.3):
    vivit = ViViT(pretrain_pth=pretrain_pth, weights_from='kinetics',
                  img_size=224,
                  num_frames=16,
                  attention_type='fact_encoder',
                  dropout_p=drop_out)

    cls_head = ClassificationHead(num_classes=num_class, in_channels=768)

    model = nn.Sequential(vivit, cls_head)
    return model.to(device)


def concat_train_dataset(train_dataset,aug_size,aug_transform,temporal_sample,path):
    #prepare list for all training data
    train_list=[train_dataset]
    #instantiate multiple dataset object with augmented data
    for _ in range(aug_size):
            aug_train_dataset = DogDataset(path, transform=aug_transform, temporal_sample=temporal_sample)
            train_list.append(aug_train_dataset)

    return utils.data.ConcatDataset(train_list)

# load the dataset for Dataloader later on
def load_dataset(
        train_ann_path,
        val_ann_path,
        aug_size=1,
        data_statics='kinetics',
        objective='supervised',
        img_size=224,
        auto_augment=None,
        num_frames=16,
        frame_interval=16,
        hflip=0.8,
        ):
    color_jitter = 0.4
    scale = None

    # to define the normalization parameters
    if data_statics == 'imagenet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif data_statics == 'kinetics':
        mean, std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    # prepare transformation for train and eval datasets accordingly
    temporal_sample = TemporalRandomCrop(
        num_frames * frame_interval)

    train_transform = transforms_train_dog(img_size=img_size,
                    augmentation=False,
                    crop_pct=None,
					 color_jitter=color_jitter,
					 auto_augment=auto_augment,
					 interpolation='bicubic',
					 mean=mean,
					 std=std,)
    train_dataset = DogDataset(train_ann_path, transform=train_transform, temporal_sample=temporal_sample)
    # to implement additional augmentation
    if aug_size>0:
        
        #prepare augmentation transform
        aug_train_transform= transforms_train_dog(img_size=img_size,
                        augmentation=True,
                        crop_pct=None,
                        hflip=hflip, # 0 for non-augment data
                        color_jitter=color_jitter,
                        auto_augment=auto_augment,
                        interpolation='bicubic',
                        mean=mean,
                        std=std,)
        
        train_dataset=concat_train_dataset(train_dataset,aug_size,aug_train_transform,temporal_sample,train_ann_path)


    val_transform = create_video_transform(
        input_size=img_size,
        is_training=False,
        interpolation='bicubic',
        mean=mean,
        std=std)

    # use the dataset class to load in DAtaset Obj
    val_dataset = DogDataset(val_ann_path, transform=val_transform, temporal_sample=temporal_sample)
    

    return train_dataset, val_dataset


def load_DataLoader(train_dataset, val_dataset, batch_size, num_worker=0):
    train_DataLoader = utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_worker,
                                             shuffle=True,
                                             pin_memory=True)
    val_DataLoader = utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_worker,
                                           shuffle=False, )
    return train_DataLoader, val_DataLoader


def training_loop(model, train_loader, val_loader, epochs, optimizer,lr_sched, criterion, save_path,clip_value, step_log=10):
    # may add accelerator ....

    progress_bar = tqdm(range(epochs * len(train_loader)))

    for ep in range(epochs):
        model.train()
        total_loss, train_total, train_correct = 0, 0, 0
        for step, batch in enumerate(train_loader):
            inputs, labels = batch  # input image already (B T C H W)/ tokenizing will be done inside Vivit model
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # BP and optimize
            optimizer.zero_grad()
            loss.backward()
            #to clip the parameters grad by values
            if clip_value>0:
                nn.utils.clip_grad_value_(model.parameters(),clip_value)
            # update parameters
            optimizer.step()

            # calculate metrices
            batch_loss = loss.item()
            total_loss += batch_loss
            logits = torch.argmax(outputs.detach(), 1).cpu()
            labels = labels.cpu()
            batch_correct = torch.sum(logits == labels).item()
            train_correct += batch_correct
            train_total += labels.size(0)

            #lr scheduler update
            lr_sched.step()
            # Monitor Progress
            if step % step_log == 0:  # print accuracy and loss every 10 steps
                print("current progress are ep{}: {}/{}".format(ep,step+1,len(train_loader)))
                print("the training loss for this batch:{}".format(batch_loss))
                print("the training accuracy for this batch: {}".format(batch_correct / labels.size(0)))
            progress_bar.update(1)

        # Save model and monitor result for this ep
        print(" epoch{}: average accuracy:{}; total loss:{}".format(ep + 1, train_correct / train_total, total_loss))
        saved_path=os.path.join(save_path, "ep{}_saved".format(ep + 1))
        torch.save(model.state_dict(), saved_path)
        print('Model saved in {}'.format(saved_path))

        # to do eval per epoch end
        print("Evaluation for ep{} start!".format(ep + 1))
        model.eval()
        eval_correct, eval_total, eval_loss = 0, 0, 0
        with torch.no_grad():
            for eval_step, eval_batch in enumerate(val_loader):
                eval_inputs, eval_labels = eval_batch
                eval_inputs = eval_inputs.to(device)
                eval_labels = eval_labels.to(device)

                preds = model(eval_inputs)
                eval_loss += criterion(preds, eval_labels).item()
                eval_labels = eval_labels.cpu()
                eval_correct += torch.sum(torch.argmax(preds.detach(), 1).cpu() == eval_labels).item()
                eval_total += eval_labels.size(0)

                print("Eval Progress:{}/{}".format(eval_step + 1, len(val_loader)))
        eval_accuracy = eval_correct / eval_total
        print("the eval accuracy:{}; eval loss:{}".format(eval_accuracy, eval_loss))


if __name__ == "__main__":
    # to add in parser for hyperparameters
    ep=10
    clip_value=1 # 0 for disabling grad clip by value


    # load in Vivit and Class_Head
    model = load_model('./vivit_model.pth')
    # load in preprocessed Dataset
    train_dataset, val_dataset = load_dataset('./data/train.csv',
                                              './data/eval.csv')
    # load them to Data Loader
    train_DataLoader, val_DataLoader = load_DataLoader(train_dataset, val_dataset, 4)

    #define optimizer and loss function
    #optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=0.005, weight_decay=0.05)
    #
    optimizer = optim.SGD(model.parameters(), momentum=0.9, nesterov=True,
                          lr=0.0005, weight_decay=0.05)
    lr_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=1, eta_min=1e-6,last_epoch=-1)
    criterion = nn.CrossEntropyLoss()

    #path for saving the model
    PATH = "./saved_model"  # folder to save the model
    os.makedirs(PATH,exist_ok=True)

    #Finetuning for the Vivit Classifier for Dog video emotions
    training_loop(model, train_DataLoader, val_DataLoader, ep, optimizer,lr_sched, criterion, PATH,clip_value, step_log=5)

