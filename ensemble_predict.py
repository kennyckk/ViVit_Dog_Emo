import torch 
import os

from video_transformer import ViViT
from transformer import ClassificationHead
from data_transform import create_video_transform, TemporalRandomCrop
from dataset import DogDataset,skip_bad_collate
from train_utils import multi_views_model

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(path,pretrain_pth=None,num_class=1,num_frames=16,input_batchNorm=False ):

    vivit = ViViT(pretrain_pth=pretrain_pth, weights_from=None,
                  img_size=224,
                  num_frames=num_frames,
                  attention_type='fact_encoder',
                  dropout_p=0)

    cls_head = ClassificationHead(num_classes=num_class, in_channels=768)
    modules=[vivit,cls_head]
    modules=torch.nn.Sequential(*modules)
    
    if input_batchNorm:#activate for input normalization along temporal axis
        modules=[torch.nn.BatchNorm3d(num_frames)]+[modules]
        modules = torch.nn.Sequential(*modules)

    model=modules
    missing, unexpected=model.load_state_dict(torch.load(path),strict=False)
    print("missing keys:{}, unexpected:{}".format(missing,unexpected))
    return model.to(device)

def eval(model,val_loader,criterion,multi_view=False):
    '''return logits predicted from each batch for each model'''
    model.eval()
    eval_correct, eval_total, eval_loss = 0, 0, 0
    logits_batches=[]
    labels=[]
    with torch.no_grad():
        for eval_step, eval_batch in enumerate(val_loader):
            eval_inputs, eval_labels = eval_batch
            eval_inputs = eval_inputs.to(device)
            eval_labels = eval_labels.to(device)
            #print(eval_labels.size())
            preds = model(eval_inputs) if not multi_view else multi_views_model(model,eval_inputs)
            
            if isinstance(criterion,torch.nn.BCEWithLogitsLoss): #special handling for BCE
                preds=torch.squeeze(preds,-1)# size is (batchsize,)
                preds=preds.detach()
                pred_logits=torch.zeros(eval_labels.size())
                pred_logits[torch.nn.functional.sigmoid(preds.cpu())>0.5]=1
                eval_labels=eval_labels.float()
            else:
                pred_logits = torch.argmax(preds.detach(), 1).cpu()


            eval_loss += criterion(preds, eval_labels).item()
            eval_labels = eval_labels.cpu()
            eval_correct += torch.sum(pred_logits == eval_labels).item()
            eval_total += eval_labels.size(0)

            print("Eval Progress:{}/{}".format(eval_step + 1, len(val_loader)))
            logits_batches.append(preds.cpu())
            labels.append(eval_labels)
    eval_accuracy = eval_correct / eval_total
    
    print("individual model eval accuracy:{}; total eval loss:{}".format(eval_accuracy, eval_loss))
    return torch.cat(logits_batches), torch.cat(labels)

def load_eval_loader(path,batch_size,num_workers=0,all_frames=16):
    val_transform = create_video_transform(
            input_size=224,
            is_training=False,
            interpolation='bicubic',
            mean=(0.45, 0.45, 0.45), 
            std=(0.225, 0.225, 0.225))
    
    temporal_sample = TemporalRandomCrop(16 * 16, full_length=True)

    val_dataset = DogDataset(path, transform=val_transform, temporal_sample=temporal_sample,num_frames=16,all_frames=all_frames)
    val_DataLoader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=False,
                                           collate_fn=skip_bad_collate )
    return val_DataLoader
    
    

def ensembles_eval(save_path,val_loader,all_frames=16):
    multi_view=True if all_frames>16 else False
    model_weights= os.listdir(save_path) #a list of trained weights.pth
    criterion = torch.nn.BCEWithLogitsLoss()
    all_logits=[]

    for idx,weight in enumerate(model_weights):
        path=os.path.join(save_path,weight)
        model=load_model(path)
        print(f"model{idx} sucessfully loaded")
        logits, labels=eval(model,val_loader,criterion,multi_view=multi_view)
        all_logits.append(logits)
    
    avg_logits=torch.sum(torch.stack(all_logits),dim=0) #torch.mean(torch.stack(all_logits),dim=0)
    #print(avg_logits)
    predictions=torch.zeros_like(avg_logits)
    predictions[torch.nn.functional.sigmoid(avg_logits)>0.5]=1
    corret=torch.sum(predictions==labels)
    accuracy=corret/labels.size(0)
    print('the ensembles accuracy is {}'.format(accuracy))
    
if __name__=="__main__":
    all_frames=48
    val_loader=load_eval_loader('./face_data/eval.csv',4,all_frames=all_frames)
    ensembles_eval('saved_model/',val_loader,all_frames=all_frames)

    

