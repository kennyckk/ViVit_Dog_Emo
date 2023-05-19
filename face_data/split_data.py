import pandas as pd
from sklearn.model_selection import train_test_split




all_data=pd.read_csv('./all_data.csv',header=0,)

data=all_data.loc[:,all_data.columns!='label']
labels=all_data['label']

#stratify to make sure balanced split in each class
train_data,test_data,train_label,test_label=train_test_split(data, labels,test_size=0.2,random_state=2,stratify=labels)

#just to check the distribution of the class after split
#print(train_label.groupby(train_label).count())

#concat back the columns from spliting
train_split=pd.concat([train_data,train_label],axis=1)
test_split=pd.concat([test_data,test_label],axis=1)
print(train_split.groupby(['label']).count())
print(test_split.groupby(['label']).count())
#save it in the curr dir for data preprocessing
train_split.to_csv('./train.csv',sep=',',index=False)
test_split.to_csv('./eval.csv',sep=',',index=False)