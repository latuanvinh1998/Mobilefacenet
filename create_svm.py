from torchvision import datasets, transforms
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from Facenet.load_dataset import load_dataset
from Model.MobileFacenet import MobileFaceNet

import torch.nn.functional
import torch
import pickle
import math
import os
import cv2
import numpy as np

def create_svm():
  model = MobileFaceNet(512).to(torch.device("cuda:0"))
  model.load_state_dict(torch.load('../PretrainedModel/model.pth'))

  dataset = load_dataset('../Dataset/Processed/')
  images = []
  labels = []

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

  for class_name in dataset:
    for path in class_name.paths:
      img = cv2.imread(path)
      img = cv2.resize(img,(112,112))
      img = transform(img)
      img = img.type(torch.FloatTensor)
      images.append(img)
      labels.append(class_name.name)

  img_batch = torch.utils.data.DataLoader(images, batch_size=32)
  labels = np.array(labels)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

  #---------------------CREATE EMBEDDING AND LABEL-----------------------------------------
  labels_encoder = LabelEncoder().fit(labels)
  labelsNum = labels_encoder.transform(labels)
  nClasses = len(labels_encoder.classes_)
  nrof_img = len(labelsNum)
  emb = np.zeros((nrof_img,512))
  idx = 0

  model.eval()

  for batch in iter(img_batch):
    with torch.no_grad():
      batch = batch.to(torch.device("cuda:0"))
      embedding = model(batch).cpu()
    emb[idx:idx+32,:] = embedding
    idx += 32

  clf = SVC(C=1, kernel='linear', probability=True)
  clf.fit(emb,labelsNum)

  fname = '../PretrainedModel/classifier.pkl'
  with open(fname, 'wb') as f:
    pickle.dump((labels_encoder, clf), f)
