import torch
from torch import optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import pandas as pd
from torch.nn import CrossEntropyLoss
import os
import matplotlib.pyplot as plt
from model import *
from tqdm import tqdm
import cv2
import numpy as np
 
from datetime import datetime
from evalsource import evaluate, get_val_pair



model_path = "../model/"
log_dir = "../LOG/"
name = str(datetime.now())[:-10].replace(' ','-').replace(':','-')






os.makedirs(os.path.dirname(model_path), exist_ok=True)
os.makedirs(os.path.dirname(log_dir+name+'/'), exist_ok=True)



writer = SummaryWriter(log_dir+name+'/')




###### PREPARE DATA ######
transform = transforms.Compose([transforms.Resize(112), transforms.CenterCrop(112), 
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

dataset = datasets.ImageFolder('../imgs/', transform=transform)
class_num = dataset[-1][1] + 1
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)





model = MobileFaceNet(512).to(torch.device("cuda:0"))
arc = Arcface(embedding_size=512, classnum=class_num).to(torch.device("cuda:0"))

paras_only_bn, paras_wo_bn = separate_bn_paras(model)
optimizer = optim.SGD([
	            {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
	            {'params': [paras_wo_bn[-1]] + [arc.kernel], 'weight_decay': 4e-4},
	            {'params': paras_only_bn}
	        ], lr = 1e-3, momentum = 0.9)

###### LOAD MODEL ######



model.load_state_dict(torch.load('../model/2020-11-19-16-13/model_accuracy:91.8.pth'))
arc.load_state_dict(torch.load('../model/2020-11-19-16-13/arc_accuracy:91.8.pth'))	
optimizer.load_state_dict(torch.load('../model/2020-11-19-16-13/optimizer_accuracy:91.8.pth'))



###### LOAD LFW IMG ######

lfw, lfw_issame = get_val_pair('data', 'lfw')


acc, std, best = evaluate(model=model, carray=lfw, issame=lfw_issame)
print("accuracy: %1.3f" %(acc*100))

model.train()

f = open('step.txt', 'r')

global_step = int(f.readline())
epoch = int(f.readline())

while epoch < 500:
	for img, label in iter(dataloader):

		img = img.to(torch.device("cuda:0"))
		label = label.to(torch.device("cuda:0"))

		optimizer.zero_grad()

		embedding = model(img)
		theta = arc(embedding, label)

		loss = CrossEntropyLoss()(theta, label)
		loss.backward()
		optimizer.step()

		if global_step%100 == 0:

			f = open('step.txt', 'w')
			f.write(str(global_step) + '\n' + str(epoch))
			f.close()

			writer.add_scalar('loss', loss, global_step)

			print("Global step: %.d ==== Epoch: %d ==== Loss: %.3f" % (global_step, epoch, loss))

		if global_step%10000 == 0 and global_step != 0:

			acc, std, best = evaluate(model=model, carray=lfw, issame=lfw_issame)

			acc = np.round(acc*100, 3)
			std = np.round(std*100, 3)

			print("Accuracy: {}+-{}".format(acc, std))
			print("Best threshold: {}".format(best))				

			name = str(datetime.now())[:-10].replace(' ','-').replace(':','-')

			model_save = os.path.join(model_path, name +'/')
			os.makedirs(os.path.dirname(model_save), exist_ok=True)

			torch.save(model.state_dict(), model_save + 'model_accuracy:{}.pth'.format(acc))
			torch.save(arc.state_dict(), model_save + 'arc_accuracy:{}.pth'.format(acc))
			torch.save(optimizer.state_dict(), model_save + 'optimizer_accuracy:{}.pth'.format(acc))

			writer.add_scalar('accuracy', acc, global_step)

			model.train()

		global_step += 1
	epoch += 1


