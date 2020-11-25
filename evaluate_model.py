import torch
import pandas as pd
import os
import cv2
import numpy as np
import bcolz
from datetime import datetime
from Facenet.evaluate import eval
import matplotlib.pyplot as plt
from Model.MobileFacenet import MobileFaceNet


def get_val_pair(path, name):
	carray = bcolz.carray(rootdir = path+name, mode='r')
	issame = np.load(path + '{}_list.npy'.format(name))
	return carray, issame


def evaluate(model, carray, issame, nrof_folds = 5):
	idx = 0
	embeddings = np.zeros([len(carray), 512])
	model.eval()
	with torch.no_grad():
		while (idx + 10) <= (len(carray)):
			batch = torch.tensor(carray[idx:idx + 10])
			emb = model(batch.to(torch.device("cuda:0"))).cpu()
			embeddings[idx:idx + 10] = emb
			idx += 10
		if idx < len(carray):
			batch = torch.tensor(carray[idx:])            
			embeddings[idx:] = model(batch.to(torch.device("cuda:0"))).cpu()
	tpr, fpr, accuracy = eval(embeddings=embeddings, actual_issame=issame, nrof_folds=nrof_folds)
	return accuracy.mean()


lfw, lfw_issame = get_val_pair('../Dataset/faces_emore/', 'lfw')

model = MobileFaceNet(512).to(torch.device("cuda:0"))
model.load_state_dict(torch.load('../PretrainedModel/model.pth'))

acc = evaluate(model=model, carray=lfw, issame=lfw_issame)
print("accuracy: %1.3f" %(acc*100))

