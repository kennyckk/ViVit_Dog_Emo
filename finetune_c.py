import os
import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from torchmetrics import Accuracy
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from video_transformer import ViViT
from transformer import ClassificationHead
from data_transform import create_video_transform, TemporalRandomCrop,transforms_train_dog,transforms_eval
from dataset import DogDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#freezing function for parameters
def freeze_layers(model,freeze_map=None):
    if freeze_map ==None:
        for _, param in model.named_parameters():
            param.requires_grad=False
    return model

#add drop out rate in different transformer layer
def drop_out_loop(model,drop_out):
    no_drop=("drop_after_time", "drop_after_pos")
    for _ , m in enumerate(model.named_modules()):
        path=m[0]
        component=m[1]
        if isinstance(component,nn.Dropout) and path not in no_drop:
            component.p=drop_out
            #component.inplace=True


# Function to load in model
def load_model(pretrain_pth, num_class=2,drop_out=0.2,freeze=False,num_frames=16,input_batchNorm=False ):
    vivit = ViViT(pretrain_pth=pretrain_pth, weights_from='kinetics',
                  img_size=224,
                  num_frames=num_frames,
                  attention_type='fact_encoder',
                  dropout_p=drop_out)
    # to change the activate the drop out layer in each transformer layer
    # if drop_out>0:
    #     drop_out_loop(vivit,drop_out)
    
    if freeze:
        vivit=freeze_layers(vivit)

    cls_head = ClassificationHead(num_classes=num_class, in_channels=768)
    modules=[vivit,cls_head]
    if input_batchNorm:#activate for input normalization along temporal axis
        modules=[nn.BatchNorm3d(num_frames)]+modules

    model = nn.Sequential(*modules)
    return model.to(device)


def concat_train_dataset(train_dataset,aug_size,temporal_sample,path,rotate,hflip,noise,mean,std,num_frames,
img_size=224,color_jitter=None,):
    #prepare list for all training data
    train_list=[train_dataset]
    #gradually incrase the chance of more augmentation with more augmentation size
    rotate_values=np.linspace(rotate,1,aug_size) 
    hflip_values=np.linspace(hflip,1,aug_size)
    noise_values=np.linspace(noise,0.5,aug_size)

    #instantiate multiple dataset object with augmented data
    for i in range(aug_size):
            aug_train_transform= transforms_train_dog(img_size=img_size,
                        augmentation=True,
                        crop_pct=None,
                        color_jitter=color_jitter,
                        auto_augment=auto_augment,
                        interpolation='bicubic',
                        mean=mean,
                        std=std,
                        rotate=rotate_values[i],
                        hflip=hflip_values[i], # 0 for non-augment data
                        noise=0)#noise_values[i])

            aug_train_dataset = DogDataset(path, transform=aug_train_transform, temporal_sample=temporal_sample,num_frames=num_frames)
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
        frame_interval=8,
        hflip=0.8,
        noise=0.2,
        rotate=0.3
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
    train_dataset = DogDataset(train_ann_path, transform=train_transform, temporal_sample=temporal_sample,num_frames=num_frames)
    # to implement additional augmentation
    if aug_size>0:
        
        #train_dataset,aug_size,temporal_sample,path,rotate,hflip,noise
        #concatenate more augmented dataset
        train_dataset=concat_train_dataset(train_dataset,aug_size,temporal_sample,train_ann_path,rotate,hflip,noise,
        mean,std,num_frames,
        img_size=img_size,
        color_jitter=color_jitter)


    val_transform = create_video_transform(
        input_size=img_size,
        is_training=False,
        interpolation='bicubic',
        mean=mean,
        std=std)

    # use the dataset class to load in DAtaset Obj
    val_dataset = DogDataset(val_ann_path, transform=val_transform, temporal_sample=temporal_sample,num_frames=num_frames)
    

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


def plot_graph(train_loss_log,train_acc_log,eval_loss_log,eval_acc_log):
    ep=len(train_loss_log)
    x=np.linspace(1,ep,num=ep)

    fig=plt.figure()
    ax0=fig.add_subplot(121,title='loss')
    ax1=fig.add_subplot(122,title='accuracy')

    ax0.plot(x,train_loss_log,'bo-',label='train')
    ax0.plot(x,eval_loss_log,'ro-',label='val')
    ax1.plot(x,train_acc_log,'bo-',label='train')
    ax1.plot(x,eval_acc_log,'ro-',label='val')

    ax0.legend()
    ax1.legend()

    fig.savefig('./loss_acc_graph.jpg')

def training_loop(model, train_loader, val_loader, epochs, optimizer,lr_sched, criterion, save_path,clip_value, step_log=10):
    # may add accelerator ....

    progress_bar = tqdm(range(epochs * len(train_loader)))
    train_loss_log=[]
    train_acc_log=[]
    eval_loss_log=[]
    eval_acc_log=[]
    best_eval_loss=float('inf')

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
            with torch.autograd.set_detect_anomaly(False):
                optimizer.zero_grad()
                loss.backward()

                #to clip the parameters grad by values
                if clip_value>0:
                    nn.utils.clip_grad_value_(model.parameters(),clip_value)
                # if ep>=10:
                #     for name, params in model.named_parameters():
                #         print(name, torch.isfinite(params.grad).all())
                #         if torch.isnan(params).any():
                #             print(f"{name} is haveing NaN weight")
                
                # update parameters
                optimizer.step()

                #check NaN weights after grad update
                # if ep>=10:
                #     if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any().item():
                #         print("weights contain NaN after grad update")
                    
            # calculate metrices
            batch_loss = loss.item()
            total_loss += batch_loss
            logits = torch.argmax(outputs.detach(), 1).cpu()
            labels = labels.cpu()
            batch_correct = torch.sum(logits == labels).item()
            train_correct += batch_correct
            train_total += labels.size(0)

            #lr scheduler update (step)
            lr_sched.step()
            # Monitor Progress
            if step % step_log == 0:  # print accuracy and loss every 10 steps
                print("current progress are ep{}: {}/{}".format(ep,step+1,len(train_loader)))
                print("the training loss for this batch:{}".format(batch_loss))
                print("the training accuracy for this batch: {}".format(batch_correct / labels.size(0)))
            progress_bar.update(1)
        #lr scheduler update (ep)
        #lr_sched.step()

        # Show results of this train epoch
        train_acc=train_correct / train_total
        print(" epoch{}: average accuracy:{}; total loss:{}".format(ep + 1, train_acc, total_loss))
        # append to train_loss list for plotting
        train_loss_log.append(total_loss/len(train_loader))
        train_acc_log.append(train_acc)


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
        #to log the loss eval for this ep for plotting
        eval_loss_log.append(eval_loss/len(val_loader))
        eval_acc_log.append(eval_accuracy)
        print("the eval accuracy:{}; eval loss:{}".format(eval_accuracy, eval_loss))

        #save the model after eval and verify loss dropped:
        #only save the model when it performs better than prev eval loss
        if eval_loss_log[-1]<best_eval_loss:
            best_eval_loss=eval_loss_log[-1]
            saved_path=os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), saved_path)
            print('Model saved in {}'.format(saved_path))
            print(f'the model saved obtained in ep {ep+1}')
        
        if ep>=10:
            plot_graph(train_loss_log,train_acc_log,eval_loss_log,eval_acc_log)
    
    return train_loss_log,train_acc_log,eval_loss_log,eval_acc_log


if __name__ == "__main__":
    torch.manual_seed(123)
    np.random.seed(123)

    # to add in parser for hyperparameters
    ep=30
    clip_value=1 # 0 for disabling grad clip by value
    noise=0
    lr=0.0005
    auto_augment=False
    freeze=False
    weight_decay=0.05 #0.05 for original
    T_0=4 # optim in step wise 
    drop_out=0 #i.e. no transfrom layer drop out/ only embed drop out
    aug_size=3
    frame_interval=8 #tune samller for more randomness in temproal sampling
    num_frames=16 #strictly 16 and cant change due to pre-trained Vivit K400
    batch_size=4
    momentum=0.9
    nesterov=True
    input_batchNorm=False

    # load in Vivit and Class_Head
    model = load_model('./vivit_model.pth',freeze=freeze,drop_out=drop_out,num_frames=num_frames,input_batchNorm=input_batchNorm)
    parameters= filter(lambda p: p.requires_grad,model.parameters()) #only need those trainable params
    #print(parameters)
    # load in preprocessed Dataset
    train_dataset, val_dataset = load_dataset('./face_data/train.csv',
                                              './face_data/eval.csv',
                                              noise=noise,
                                              auto_augment=auto_augment,
                                              aug_size=aug_size,
                                              frame_interval=frame_interval,
                                              num_frames=num_frames)
    # load them to Data Loader
    train_DataLoader, val_DataLoader = load_DataLoader(train_dataset, val_dataset, batch_size=batch_size)

    #define optimizer and loss function
    #optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=weight_decay)
    #
    optimizer = optim.SGD(parameters, momentum=momentum, nesterov=nesterov,
                         lr=lr, weight_decay=weight_decay)
    lr_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=1e-6,last_epoch=-1)
    #lr_sched=None
    criterion = nn.CrossEntropyLoss()

    #path for saving the model
    PATH = "./saved_model"  # folder to save the model
    os.makedirs(PATH,exist_ok=True)

    #Finetuning for the Vivit Classifier for Dog video emotions
    train_loss_log,train_acc_log,eval_loss_log,eval_acc_log=training_loop(model, 
                                                                            train_DataLoader, 
                                                                            val_DataLoader, 
                                                                            ep, 
                                                                            optimizer,
                                                                            lr_sched, 
                                                                            criterion, 
                                                                            PATH,
                                                                            clip_value, 
                                                                            step_log=5)

    #plotting
    plot_graph(train_loss_log,train_acc_log,eval_loss_log,eval_acc_log)
