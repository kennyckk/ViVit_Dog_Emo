import os
import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from torchmetrics import Accuracy
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import wandb

from video_transformer import ViViT
from transformer import ClassificationHead
from data_transform import create_video_transform, TemporalRandomCrop,transforms_train_dog,transforms_eval
from dataset import DogDataset,skip_bad_collate
from train_utils import Save_Multi_Models, multi_views_model

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
def load_model(pretrain_pth, custom_weights=None,num_class=2,drop_out=0.2,freeze=False,num_frames=16,input_batchNorm=False ):
    weights_from="kinetics" if pretrain_pth!= None else None

    vivit = ViViT(pretrain_pth=pretrain_pth, weights_from=weights_from,
                  img_size=224,
                  num_frames=num_frames,
                  attention_type='fact_encoder',
                  dropout_p=0)
    # to change the activate the drop out layer in each transformer layer
    if drop_out>0:
        drop_out_loop(vivit,drop_out)
    
    if freeze:
        vivit=freeze_layers(vivit)

    cls_head = ClassificationHead(num_classes=num_class, in_channels=768)
    modules=[vivit,cls_head]
    modules=nn.Sequential(*modules)
    if custom_weights:
        missing, unexpected=modules.load_state_dict(torch.load(custom_weights),strict=False)
        print("missing keys:{}, unexpected:{}".format(missing,unexpected))
    
    if input_batchNorm:#activate for input normalization along temporal axis
        modules=[nn.BatchNorm3d(num_frames)]+[modules]
        modules = nn.Sequential(*modules)

    model=modules

    return model.to(device)


def concat_train_dataset(train_dataset,aug_size,path,rotate,hflip,noise,mean,std,num_frames,
img_size=224,color_jitter=None,auto_augment=None,temporal_random=False,frame_interval=8,all_frames=128):
    #prepare list for all training data
    train_list=[train_dataset]
    #gradually incrase the chance of more augmentation with more augmentation size
    rotate_values=np.linspace(rotate,1,aug_size) 
    hflip_values=np.linspace(hflip,1,aug_size)
    noise_values=np.linspace(0,noise,aug_size) #noise is the upper bound prob

    #instantiate multiple dataset object with augmented data
    for i in range(aug_size):
            temporal_sample=TemporalRandomCrop(num_frames * frame_interval,temporal_random=temporal_random)

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
                        noise=noise_values[i])

            aug_train_dataset = DogDataset(path, transform=aug_train_transform, temporal_sample=temporal_sample,num_frames=num_frames,
            all_frames=all_frames)
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
        hflip=0.2,
        noise=0.2,
        rotate=0.2,
        temporal_random=False,
        all_frames=128
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
        num_frames * num_frames, full_length=True) #make the frame interval large enough for most videos in intial train and eval

    train_transform = transforms_train_dog(img_size=img_size,
                    augmentation=False,
                    crop_pct=None,
					 color_jitter=color_jitter,
					 auto_augment=auto_augment,
					 interpolation='bicubic',
					 mean=mean,
					 std=std,)
    train_dataset = DogDataset(train_ann_path, transform=train_transform, temporal_sample=temporal_sample,num_frames=num_frames,
    all_frames=all_frames)
    # to implement additional augmentation
    if aug_size>0:
        
        #train_dataset,aug_size,temporal_sample,path,rotate,hflip,noise
        #concatenate more augmented dataset
        train_dataset=concat_train_dataset(train_dataset,aug_size,train_ann_path,rotate,hflip,noise,
        mean,std,num_frames,
        img_size=img_size,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        temporal_random=temporal_random,
        frame_interval=frame_interval,
        all_frames=all_frames)


    val_transform = create_video_transform(
        input_size=img_size,
        is_training=False,
        interpolation='bicubic',
        mean=mean,
        std=std)

    # use the dataset class to load in DAtaset Obj
    val_dataset = DogDataset(val_ann_path, transform=val_transform, temporal_sample=temporal_sample,num_frames=num_frames,
    all_frames=all_frames)
    

    return train_dataset, val_dataset


def load_DataLoader(train_dataset, val_dataset, batch_size, num_worker=0):
    train_DataLoader = utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_worker,
                                             shuffle=True,
                                             pin_memory=True,
                                             collate_fn=skip_bad_collate)
    val_DataLoader = utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_worker,
                                           shuffle=False,
                                           collate_fn=skip_bad_collate )
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

def plot_lr_loss(lr_rate, batchwise_loss):
    
    fig,ax=plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(lr_rate,batchwise_loss)
    ax.set_title("Training Loss vs lr rate")

    fig.savefig('./lr_loss.jpg')

def training_loop(model, train_loader, val_loader, epochs, optimizer,lr_sched, criterion, save_path,clip_value, step_log=10,T_0=None,T_mul=1,
multi_view=True,save_multiples=False):
    # may add accelerator ....

    progress_bar = tqdm(range(epochs * len(train_loader)))
    save_multi_models=Save_Multi_Models(save_path)
    train_loss_log=[]
    train_acc_log=[]
    eval_loss_log=[]
    eval_acc_log=[]
    last_restart=0
    #best_eval_acc=float('-inf')
    #these 2 are to define initial lr
    #batchwise_loss=[]
    #lr_rates=[]


    for ep in range(epochs):
        model.train()
        total_loss, train_total, train_correct = 0, 0, 0
        for step, batch in enumerate(train_loader):
            inputs, labels = batch  # input image already (B T C H W)/ tokenizing will be done inside Vivit model
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(inputs) if not multi_view else multi_views_model(model,inputs)
            

            if isinstance(criterion, nn.BCEWithLogitsLoss):
                outputs=torch.squeeze(outputs,dim=-1)
                labels=labels.float()
                
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
            if isinstance(criterion,nn.BCEWithLogitsLoss): #special handling for BCE
                logits=torch.zeros(labels.size())
                logits[nn.functional.sigmoid(outputs.detach().cpu())>0.5]=1
            else:
                logits = torch.argmax(outputs.detach(), 1).cpu()
            labels = labels.cpu()
            batch_correct = torch.sum(logits == labels).item()
            train_correct += batch_correct
            train_total += labels.size(0)

            #this is to find the initial lr 
            #lr_rates.append(optimizer.param_groups[0]["lr"])
            #batchwise_loss.append(batch_loss)
            #lr scheduler update (step)
            # if lr_sched is not None:
            #     lr_sched.step()

            # Monitor Progress
            if step % step_log == 0:  # print accuracy and loss every 10 steps
                print("current progress are ep{}: {}/{}".format(ep,step+1,len(train_loader)))
                print("the training loss for this batch:{}".format(batch_loss))
                print("the training accuracy for this batch: {}".format(batch_correct / labels.size(0)))
                print("current learning rate is {}".format(optimizer.param_groups[0]["lr"]))
            progress_bar.update(1)
        
        # Show results of this train epoch
        train_acc=train_correct / train_total
        print(" epoch{}: average accuracy:{}; total loss:{}".format(ep + 1, train_acc, total_loss))
        # append to train_loss list for plotting
        train_loss_ep=total_loss/len(train_loader)
        train_loss_log.append(train_loss_ep)
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

                preds = model(eval_inputs) if not multi_view else multi_views_model(model,eval_inputs)

                if isinstance(criterion,nn.BCEWithLogitsLoss): #special handling for BCE
                    preds=torch.squeeze(preds,-1)
                    pred_logits=torch.zeros(eval_labels.size())
                    pred_logits[nn.functional.sigmoid(preds.detach().cpu())>0.5]=1
                    eval_labels=eval_labels.float()
                else:
                    pred_logits = torch.argmax(preds.detach(), 1).cpu()


                eval_loss += criterion(preds, eval_labels).item()
                eval_labels = eval_labels.cpu()
                eval_correct += torch.sum(pred_logits == eval_labels).item()
                eval_total += eval_labels.size(0)

                print("Eval Progress:{}/{}".format(eval_step + 1, len(val_loader)))
        eval_accuracy = eval_correct / eval_total
        #to log the loss eval for this ep for plotting
        eval_loss_ep=eval_loss/len(val_loader)
        eval_loss_log.append(eval_loss/len(val_loader))
        eval_acc_log.append(eval_accuracy)
        print("the eval accuracy:{}; eval loss:{}".format(eval_accuracy, eval_loss))
        #wandb to log
        experiment.log({
            'training_loss':train_loss_ep,
            'training_accuracy':train_acc,
            'learning_rate':optimizer.param_groups[0]["lr"],
            'epochs':ep,
            'eval_loss':eval_loss_ep,
            'eval accuracy':eval_accuracy,})

        #lr scheduler update (ep)
        if lr_sched is not None:
            lr_sched.step()
        #save the model after eval and verify loss dropped:
        
        #to save multiple model for ensembles only on specific ep intervals for cosine warm up lr sched
        if save_multiples==True:
            if (ep+1)% (T_0+last_restart)==0:
                save_multi_models.check_best_n(eval_accuracy,model,ep)
                if T_mul!=1:
                    last_restart+=T_0
                    T_0*=T_mul
        #only save the model when it performs better than prev eval loss
        else:
            if eval_acc_log[-1]>best_eval_acc:
                best_eval_acc=eval_acc_log[-1]
                saved_path=os.path.join(save_path, "best_model.pth")
                torch.save(model.state_dict(), saved_path)
                print('Model saved in {}'.format(saved_path))
                print(f'the model saved obtained in ep {ep+1}')
            
        plot_graph(train_loss_log,train_acc_log,eval_loss_log,eval_acc_log)
        #to plot graph for the learning rate and loss relationship
        #plot_lr_loss(lr_rates, batchwise_loss)
    save_multi_models.model_maps()
    
    return train_loss_log,train_acc_log,eval_loss_log,eval_acc_log

def optimizer_options(option,lr,parameters,momentum=0.9 ,#for SGD only 
    nesterov=True, #for SGD only
    T_0=None, # optim in step wise  for lr scheduler only
    T_mul=1,
    eta_min=None,
    ep=None,
    weight_decay=None,):

    if option=='adam':
        optimizer = optim.AdamW(parameters, betas=(0.9, 0.999), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=nesterov,lr=lr, weight_decay=weight_decay)

    #exponential LR rate
    lower_bound=eta_min
    upper_bound=lr
    gamma=np.exp(np.log(lower_bound/upper_bound)/ep)
    lr_sched =optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma,verbose=True)
    print("the lr would increment by step with {}".format(gamma))

    #lr_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mul, eta_min=eta_min,last_epoch=-1)

    #this is to find the best initial learning rate
    # lower_bound=1e-7
    # upper_bound=1e-1
    # increment=np.exp(np.log(upper_bound/lower_bound)/(T_0*ep))
    # print("the lr would increment by step with {}".format(increment))
    # lambda1= lambda step: increment**step
    # lr_sched= optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda1)

    #lr_sched=None


    return optimizer,lr_sched

if __name__ == "__main__":
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    # Basic Hyperparameters
    loss_fnc="bce" 
    pretrain_pth='./vivit_model.pth'
    custom_weights=None #'./best_model.pth'
    ep=120
    clip_value=1 # 0 for disabling grad clip by value
    lr=5e-6
    freeze=False

    #overfitting control
    noise=0.2 # upperbound for noise prob
    auto_augment=False
    weight_decay=0.1 #0.05 for original
    drop_out=0 #i.e. transfo4rm layer drop out only
    aug_size=3
    frame_interval=8 #tune samller for more randomness in temproal sampling; no use for multi views options
    temporal_random=True
    batch_size=4
    input_batchNorm=True
    num_frames=16 #strictly unchageable
    frames_limit=32 #the total frames that the model would take for one video data for multi views options
    multi_view=True #whether using multi-view training or inference

    #optimizer and lr sched
    options="sgd"
    momentum=0.9 #for SGD only 
    nesterov=True #for SGD only
    T_0=20 # optim in ep wise  for cosine warmup only
    eta_min=1e-8 # optim in ep wise  for cosine warmup only
    T_mul=1
    save_multiples=True

    experiment= wandb.init(project='ViVit_Dog_SGD',resume='allow', anonymous='allow')
    experiment.config.update(dict(
        ep=80,
        lr=2e-3,
        noise=0.1, # upperbound for noise prob
        weight_decay=0.2, #0.05 for original
        drop_out=0.5, #i.e. transform layer drop out only
        aug_size=3,
        frame_interval=8, #tune samller for more randomness in temproal sampling
        temporal_random=True,
        batch_size=4,
        input_batchNorm=True,
        optimizer="sgd",
        momentum=0.9, #for SGD only 
        nesterov=True, #for SGD only
        T_0=20, # optim in ep wise  for cosine warmup only
        eta_min=1e-5 # optim in ep wise  for cosine warmup only
        #lr_sched=None
    ))

    num_class=1 if loss_fnc=="bce" else 2

    # load in Vivit and Class_Head
    model = load_model(pretrain_pth,custom_weights=custom_weights,num_class=num_class,freeze=freeze,drop_out=drop_out,num_frames=num_frames,input_batchNorm=input_batchNorm)
    parameters= filter(lambda p: p.requires_grad,model.parameters()) #only need those trainable params
    #print(parameters)
    # load in preprocessed Dataset
    train_dataset, val_dataset = load_dataset('./face_data/train.csv',
                                              './face_data/eval.csv',
                                              noise=noise,
                                              auto_augment=auto_augment,
                                              aug_size=aug_size,
                                              frame_interval=frame_interval,
                                              num_frames=num_frames,
                                              temporal_random=temporal_random,
                                              all_frames=frames_limit)
    # load them to Data Loader
    train_DataLoader, val_DataLoader = load_DataLoader(train_dataset, val_dataset, batch_size=batch_size)

    #define optimizer and loss function
    optimizer,lr_sched = optimizer_options(options,lr,parameters,momentum=momentum ,#for SGD only 
    nesterov=nesterov, #for SGD only
    T_0=T_0, # optim in step wise  for lr scheduler only
    T_mul=T_mul,
    eta_min=eta_min,
    ep=ep,
    weight_decay=weight_decay)

    
    criterion = nn.BCEWithLogitsLoss() if loss_fnc=="bce" else nn.CrossEntropyLoss()

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
                                                                            step_log=1000,
                                                                            T_0=T_0,
                                                                            T_mul=T_mul,
                                                                            multi_view=multi_view,
                                                                            save_multiples=save_multiples
                                                                            )

    #plotting
    plot_graph(train_loss_log,train_acc_log,eval_loss_log,eval_acc_log)
