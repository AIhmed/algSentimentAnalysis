import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset,DataLoader 
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
from farasa.segmenter import FarasaSegmenter 
from arabert.preprocess_arabert import preprocess
import pandas as pd 
import numpy as np 
arabert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabert")

farasa_segmenter=FarasaSegmenter(interactive=True)

pre_trained_arabert=AutoModel.from_pretrained("aubmindlab/bert-base-arabert")


def get_correct_prediction(predictions,labels):
    return predictions.argmax(dim=1).eq(labels).sum()

class AlgerianClassifierDataset(Dataset):
	def __init__(self,comments,tokenizer,max_len,sentiment):
		self.tokenizer=tokenizer
		self.comments=comments
		self.max_len=max_len
		self.sentiment=sentiment
	def __len__(self):
		return len(self.comments)

	def __getitem__(self,item):
		comment=str(self.comments[item])
		sentiment=self.sentiment[item]
		text_preprocessed=preprocess(comment,do_farasa_tokenization=True,farasa=farasa_segmenter,use_farasapy=True)
		enconding=self.tokenizer.encode_plus(text_preprocessed,max_length=self.max_len,truncation=True,add_special_tokens=True,pad_to_max_length=True,return_attention_mask=True,return_token_type_ids=True,return_tensors='pt')
		return{
		'comment':comment,
		'input_ids':enconding['input_ids'],
		'attention_mask':enconding['attention_mask'],
		'sentiment':torch.tensor(sentiment,dtype=torch.long)
		}# we return this dictionnary because we need it later when we iterate through the batch 

class AlgerianSentimentClassifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.arabert_model=pre_trained_arabert
		self.drop=nn.Dropout(p=0.3)
		self.output=nn.Linear(self.arabert_model.config.hidden_size,3)
		self.softmax=nn.Softmax(dim=1)

	def forward(self,input_ids,attention_mask):
		hidden_output,pooled_output=self.arabert_model(input_ids=input_ids,attention_mask=attention_mask)
		t=self.drop(pooled_output)
		t=F.relu(self.output(pooled_output))
		return self.softmax(t)

#----------------------------------------------------data preparation----------------------------------------------------------------------
df=pd.read_csv("datasets/alg_sentiment_analysis.csv")
classifier_df=df.sample(frac=1).reset_index(drop=True)
endOfTrainSet=4340
startOfTestSet=4340
trainSet=classifier_df.iloc[:endOfTrainSet,:]
classifier_traindataset=AlgerianClassifierDataset(trainSet['comment'],arabert_tokenizer,256,trainSet['label'])
classifier_data_loader=DataLoader(classifier_traindataset,batch_size=10,shuffle=True)

#--------------------------------------------------training parameters ---------------------------------------------------------------------
algerianClassifier=AlgerianSentimentClassifier()
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.AdamW(algerianClassifier.parameters(),lr=2e-5)
total_steps=len(classifier_data_loader)*2
schedular=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)#schedul the change of the learning rate
#------------------------------------------------------------------------------------------------------------------------------------------

def get_iter(batch_size,comments):
	rest=len(comments)-len(comments)%batch_size
	return rest/batch_size


def train_model(model,data_loader,loss_fn,optimizer,schedular,file):
	model=model.train()
	losses=[]
	num_batch=get_iter(10,trainSet['comment'])
	correct_predictions=0.0
	batch=0
	for sample in data_loader:
		batch=batch+1
		if batch<=num_batch:
			input_ids=sample['input_ids'].reshape(10,256)
			attention_mask=sample['attention_mask'].reshape(10,256)
			sentiment=sample['sentiment']
			preds=model(input_ids,attention_mask)
			print('predicting\n\n')
			print(preds)
			correct_predictions=get_correct_prediction(preds,sentiment)+correct_predictions
			loss=loss_fn(preds,sentiment)#calculate the CrossLossEntropy
			losses.append(loss.item())
			loss.backward()#to calculate the gradients 
			nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)#clip the gradient to counter gradient explosion
			optimizer.step()#update the weights 
			schedular.step()#performs L2 regularisation 
			optimizer.zero_grad()#zero out the gradient 
			print(f"the number of correct predictions is {get_correct_prediction(preds,sentiment)}\n\n")
			print(f"{batch} batchs are done out of {num_batch}")
			print(f"{correct_predictions} out of {batch*10} comments ")
			comments_done=10*batch# number of comments already passed throught the network per epoch 
			percentage=correct_predictions/comments_done
			print(f"accuracy up until now {percentage}")
			torch.save(model.state_dict(),file)#save the model parameters per each batch 

	return correct_predictions /(num_batch*10)
'''
for epoch in range(2):
	classifier_save_file=
	print(f"epoch {epoch} out of { 2 }\n\n")
	train_acc=train_model(algerianClassifier,classifier_data_loader,loss_fn,optimizer,schedular,classifier_save_file)
	print(f"training accracy {train_acc} \n\n")
	torch.save(algerianClassifier.state_dict(),classifier_save_file)

'''
#---------------------------------------------------------testing data preparation--------------------------------------------------------
testSet = classifier_df.iloc[startOfTestSet:,:]
testSet.reset_index(drop=True,inplace=True)
classifier_testdataset=AlgerianClassifierDataset(testSet['comment'],arabert_tokenizer,256,testSet['label'])
classifier_testdata_loader=DataLoader(classifier_testdataset,batch_size=10,shuffle=True)
#-----------------------------------------------------------------------------------------------------------------------------------------

def eval_model(model,data_loader):
	model=model.eval()
	losses=[]
	num_batch=get_iter(10,testSet['comment'])
	correct_predictions=0.0
	batch=0
	sentiment_tensor=torch.tensor([])
	preds_tensor=torch.tensor([])
	with torch.no_grad():
		for sample in data_loader:
			batch=batch+1
			if batch<=num_batch:
				comment=sample['comment']
				input_ids=sample['input_ids'].reshape(10,256)
				attention_mask=sample['attention_mask'].reshape(10,256)
				sentiment=sample['sentiment']
				preds=model(input_ids,attention_mask)
				print(preds)
				correct_predictions=get_correct_prediction(preds,sentiment)+correct_predictions
				for i in range(10):
					print(f" le commentaire est {comment[i]} le sentiment pour ce {sentiment[i]} la prediction pour ce {preds.argmax(dim=1)[i]}\n\n\n")

'''
alg_loaded_classifier=AlgerianSentimentClassifier()
alg_loaded_classifier.load_state_dict(torch.load(""))
eval_model(alg_loaded_classifier,classifier_testdata_loader)
'''