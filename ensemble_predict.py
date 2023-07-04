import torch 
import os
from sklearn.metrics import f1_score, precision_score,recall_score,accuracy_score
import numpy as np


from video_transformer import ViViT
from transformer import ClassificationHead
from data_transform import create_video_transform, TemporalRandomCrop
from dataset import DogDataset,skip_bad_collate,DecordInit
from train_utils import multi_views_model

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(path,pretrain_pth=None,num_class=1,num_frames=16,input_batchNorm=True ):

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

def eval_single(model,video,multi_view=False):
    model.eval()

    with torch.no_grad():
        #the val_loader already (1,T,C,H,W) in single video case
        video=video.to(device)
        preds= model(video) if not multi_view else multi_views_model(model,video) #(1,1)
        preds= torch.squeeze(preds,-1).detach().cpu()

    return preds  #(1,)

def eval(model,val_loader,criterion,multi_view=False):
    '''return logits predicted from each batch for each model'''
    f1_score=F1_score()
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
            f1_score.record(pred_logits,eval_labels)
            eval_correct += torch.sum(pred_logits == eval_labels).item()
            eval_total += eval_labels.size(0)

            print("Eval Progress:{}/{}".format(eval_step + 1, len(val_loader)))

            logits_batches.append(preds.cpu())
            labels.append(eval_labels)

    eval_accuracy = eval_correct / eval_total
    
    f1_score.calculate()
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

def load_single_video(vid_path,all_frames):
    """to load a single video for demo/ UI and return tensor in shape (1,T,C,H,W)"""
    val_transform = create_video_transform(
        input_size=224,
        is_training=False,
        interpolation='bicubic',
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225))

    temporal_sample = TemporalRandomCrop(16 * 16, full_length=True)

    v_decorder= DecordInit()
    v_reader=v_decorder(vid_path)
    assert v_reader != None, "Video cannot be loaded propoerly"
    total_frames= len(v_reader)

    model_input_frame=16 #strictly unchanged for model input

    if all_frames is None:
        # Sampling video frames
        start_frame_ind, end_frame_ind = temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= model_input_frame #strictly unchanged for model input
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, model_input_frame, dtype=int)
        video = v_reader.get_batch(frame_indice).asnumpy()
    else:
        # this is to get all frames from the video up to max length defined
        if total_frames > all_frames:  # just get the max length that can be taken
            frame_indice = np.linspace(0, all_frames - 1, all_frames, dtype=int)
            video = v_reader.get_batch(frame_indice).asnumpy()
        # print('the video is over specified frames', video.shape)

        else:  # will need to pad the rest of the frames
            frame_indice = np.linspace(0, total_frames - 1, total_frames, dtype=int)
            video = v_reader.get_batch(frame_indice).asnumpy()  # in numpy array T,H,W,C
            v_shape = video.shape
            # print('the video is under specified frames', v_shape)
            pad_shape = (all_frames - total_frames, v_shape[1], v_shape[2], v_shape[3])  # prepare a shape for the pads
            pads = np.zeros(pad_shape)

            video = np.concatenate((video, pads), axis=0)
        # print('the video is under specified frames', video.shape)
    del v_reader

    with torch.no_grad():
        video = torch.from_numpy(video).permute(0, 3, 1, 2)
        video = val_transform(video)

    return torch.unsqueeze(video,dim=0)

class F1_score(object):
    def __init__(self):
        self.logits_batch=[]
        self.labels_batch=[]
        self.F1_score=[]
    
    def record(self,pred_logits,eval_labels):
        self.logits_batch.append(pred_logits)
        self.labels_batch.append(eval_labels)

    def calculate(self):
        labels=torch.cat(self.labels_batch)
        logits=torch.cat(self.logits_batch)
        print(labels.size(),logits.size())
        # unique_cls= torch.unique(labels)
        # for cls in unique_cls:
        #     cls_num=labels[labels==cls].size(0)
        #     correct=torch.sum(logits[labels==cls]==cls).item()
        #     F1_score_cls=correct/cls_num
        #     print(f"the F1 score for this class {cls} is:",F1_score_cls)
        #     self.F1_score.append(F1_score_cls)
        # print("the F1 Score is:",sum(self.F1_score)/len(self.F1_score))
        print("the true F1 Score is ", f1_score(labels, logits))
        print("the true precision Score is ", precision_score(labels, logits))
        print("the true recall Score is ", recall_score(labels, logits))
        print("the Accuracy Score is ", accuracy_score(labels, logits))
def multi_model_scores(avg_logits, labels):
    #print(avg_logits)
    predictions=torch.zeros_like(avg_logits)
    #print(torch.nn.functional.sigmoid(avg_logits))
    predictions[torch.nn.functional.sigmoid(avg_logits)>0.5]=1
    corret=torch.sum(predictions==labels)
    accuracy=corret/labels.size(0)

    f1_score=F1_score()
    f1_score.record(predictions,labels)
    f1_score.calculate()
    print('the ensembles accuracy is {}'.format(accuracy))

def ensembles_eval(save_path,val_loader,all_frames=16,score=True):
    all_frames=0 if all_frames is None else all_frames
    multi_view=True if all_frames>=16 else False
    model_weights= os.listdir(save_path) #a list of trained weights.pth
    criterion = torch.nn.BCEWithLogitsLoss()
    all_logits=[]

    for idx,weight in enumerate(model_weights):
        path=os.path.join(save_path,weight)
        model=load_model(path)
        print(f"model{idx} sucessfully loaded")
        if score:
            logits, labels=eval(model,val_loader,criterion,multi_view=multi_view)
        else:
            logits= eval_single(model,val_loader,multi_view=multi_view)
        all_logits.append(logits)
        #print(torch.nn.functional.sigmoid(logits))

    avg_logits=torch.sum(torch.stack(all_logits),dim=0) #combine logits from ensemble models
    if score: multi_model_scores(avg_logits,labels)
    return (avg_logits,labels) if score else (avg_logits,None)

def inference(mode,vid_path=None, score=True):
    if not score :assert vid_path!=None, "must input single vid path if for end user usage"

    if mode=='average':
        all_frames=[32]
        dir=['final_model/average/']
    else:
        all_frames = [None, 80]
        dir = ['final_model/maximum/16/', 'final_model/maximum/80/']

    logits = []
    for id, case in enumerate(all_frames):
        val_loader = load_eval_loader('./face_data/eval.csv', 1, all_frames=case) if score else \
            load_single_video(vid_path,all_frames=case)

        logits_labels = ensembles_eval(dir[id], val_loader, all_frames=case, score=score) #output is tuple (avg_logits, labels) if doing scoring
        logits.append(logits_labels[0])
    logits = torch.sum(torch.stack(logits), dim=0)
    if score:
        multi_model_scores(logits, logits_labels[1])
    else:
        return 1 if torch.nn.functional.sigmoid(logits)>=0.5 else 0



if __name__=="__main__":
    mode='max' #average or maximum
    prediction=inference(mode,'./static/c2.mp4' ,score=False)
    print(prediction)
    # all_frames = [32,48,64,80]
    # dir = ['final_model/maximum/80/']*4
    #
    # logits = []
    # for id, case in enumerate(all_frames):
    #     val_loader = load_eval_loader('./face_data/eval.csv', 1, all_frames=case)
    #     avg_logits, labels = ensembles_eval(dir[id], val_loader, all_frames=case)
    #     logits.append(avg_logits)
    # logits = torch.sum(torch.stack(logits), dim=0)
    # multi_model_scores(logits, labels)

    

