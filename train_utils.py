import torch
import os

class Save_Multi_Models(object):
    def __init__(self,save_path,best_n=3):
        self.best_eval_accs=[]
        self.check_points=[]
        self.save_path=save_path
        self.best_n=best_n
    
    def save_model (self,model,slot):
        path=os.path.join(self.save_path,"best_model{}.pth".format(slot))
        torch.save(model.state_dict(),path)
        print('Model saved in {}'.format(path))
        

    def check_best_n(self, curr_eval,model,ep):
        if len(self.best_eval_accs)<self.best_n: #they will always be the best n
            self.best_eval_accs.append(curr_eval)
            self.check_points.append(ep) #to record which ep this model saved from
            slot=len(self.best_eval_accs) #i.e. [best_model1, best_model2, etc]
            self.save_model(model,slot)
        else:
            min_acc= min(self.best_eval_accs)
            min_acc_idx= self.best_eval_accs.index(min_acc)
            
            if curr_eval>min_acc:
                self.best_eval_accs[min_acc_idx]=curr_eval
                self.check_points[min_acc_idx]=ep
                slot=min_acc_idx+1
                self.save_model(model,slot)
            
    def model_maps(self): #to create a txt file to document the saved model acc and ep
        path= os.path.join(self.save_path,'model_info.txt')
        with open(path, 'w') as f:
            accs=str(self.best_eval_accs)
            eps=str(self.check_points)
            f.writelines(accs)
            f.write('\n')
            f.writelines(eps)

def multi_views_model(model,inputs):
    #the inputs would be B,T,C,H,W--> BT',16,C,H,W
    shape=inputs.size()
    inputs=inputs.reshape(-1,16,shape[2],shape[3],shape[4])

    output=model(inputs) # (BT',1) logits
    output=output.reshape(shape[0],-1,1) 
    output= torch.mean(output, dim=1)
    #print(output.size())
    return output

        
        
            

