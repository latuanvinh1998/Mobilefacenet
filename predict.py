from torchvision import datasets, transforms

import numpy as np 
import os
import sys
import torch
import pickle
import cv2
import time
import argparse
import math
from Model.MobileFacenet import MobileFaceNet
sys.path.insert(1, "MTCNN")
from mtcnn import MTCNN
from detect_face import extract_face


FONT = cv2.FONT_HERSHEY_SIMPLEX 

SVM_MODEL =  "../PretrainedModel/classifier.pkl"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(image_size=112, device=device)

model = MobileFaceNet(512).to(torch.device("cuda:0"))
model.load_state_dict(torch.load('../PretrainedModel/model.pth'))

with open(SVM_MODEL, 'rb') as infile:
	(class_name, svm_model) = pickle.load(infile)
	
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def ImageMain(args):
	img = cv2.imread(args.image_path)

	boxes = detector.detect(img=img, select_largest=args.select_largest, proba=False, landmarks=False)
	if boxes is not None:
		for box in boxes:
			cv2.rectangle(img, tuple((np.int32(box[0]), np.int32(box[1]))), tuple((np.int32(box[2]), np.int32(box[3]))), (255, 255, 255), 1)
			face = extract_face(img, box, image_size=160)
			face = cv2.resize(face,(112,112))
			face = transform(face)
			face = face.type(torch.FloatTensor)
			face = torch.unsqueeze(face, 0)
			model.eval()
			with torch.no_grad():
			  face = face.to(torch.device("cuda:0"))
			  emb_buff = model(face).cpu()
			buff = emb_buff.numpy().reshape(1, -1)
			predictions = svm_model.predict_proba(buff).ravel()
			best_class_idxs = np.argmax(predictions)
			confidence = predictions[best_class_idxs]*100
			name = class_name.inverse_transform([best_class_idxs])
			predict_str = '%s: %.3f'%(name, confidence)
			print(predict_str)
			pos = tuple((np.int32(box[0]), np.int32(box[1])))
			cv2.putText(img, predict_str, pos, FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.imshow('preview',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def VideoMain(args):


	if args.MODE == 'VIDEO':
		cap = cv2.VideoCapture(args.video_path)
	elif args.MODE == 'WEBCAM': 
		cap = cv2.VideoCapture(0)

	if not (cap.isOpened()):
		print("Could not open video device")

	while(True): 
		start = time.time()
		ret, frame = cap.read()
		img = cv2.flip(frame, 1)
		boxes, points = detector.detect(img, select_largest=args.select_largest, proba=False, landmarks=True)

		if boxes is not None:
			for box in boxes:
				cv2.rectangle(img, tuple((np.int32(box[0]), np.int32(box[1]))), tuple((np.int32(box[2]), np.int32(box[3]))), (255, 255, 255), 1)
				face = extract_face(img, box, image_size=160)
				face = cv2.resize(face,(112,112))
				face = transform(face)
				face = face.type(torch.FloatTensor)
				face = torch.unsqueeze(face, 0)
				model.eval()
				with torch.no_grad():
				  face = face.to(torch.device("cuda:0"))
				  emb_buff = model(face).cpu()
				buff = emb_buff.numpy().reshape(1, -1)
				predictions = svm_model.predict_proba(buff).ravel()
				best_class_idxs = np.argmax(predictions)
				confidence = predictions[best_class_idxs]*100
				name = class_name.inverse_transform([best_class_idxs])
				predict_str = '%s: %.3f'%(name, confidence)
				print(predict_str)
				pos = tuple((np.int32(box[0]), np.int32(box[1])))
				cv2.putText(img, predict_str, pos, FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
		fps_str = 'fps: %d'%(1/(time.time() - start))
		cv2.putText(img, fps_str, (10, 50), FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.imshow('preview',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

def Recognition(img):

	# FONT = cv2.FONT_HERSHEY_SIMPLEX 

	# SVM_MODEL =  "../PretrainedModel/classifier.pkl"

	# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	# detector = MTCNN(image_size=112, device=device)

	# model = MobileFaceNet(512).to(torch.device("cuda:0"))
	# model.load_state_dict(torch.load('../PretrainedModel/model.pth'))

	# with open(SVM_MODEL, 'rb') as infile:
	# 	(class_name, svm_model) = pickle.load(infile)
		
	# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
	
	boxes, points = detector.detect(img, select_largest=True, proba=False, landmarks=True)

	if boxes is not None:
		for box in boxes:
			cv2.rectangle(img, tuple((np.int32(box[0]), np.int32(box[1]))), tuple((np.int32(box[2]), np.int32(box[3]))), (255, 255, 255), 1)
			face = extract_face(img, box, image_size=160)
			face = cv2.resize(face,(112,112))
			face = transform(face)
			face = face.type(torch.FloatTensor)
			face = torch.unsqueeze(face, 0)
			model.eval()
			with torch.no_grad():
			  face = face.to(torch.device("cuda:0"))
			  emb_buff = model(face).cpu()
			buff = emb_buff.numpy().reshape(1, -1)
			predictions = svm_model.predict_proba(buff).ravel()
			best_class_idxs = np.argmax(predictions)
			confidence = predictions[best_class_idxs]*100
			name = class_name.inverse_transform([best_class_idxs])
			predict_str = '%s: %.3f'%(name, confidence)
			print(predict_str)
			pos = tuple((np.int32(box[0]), np.int32(box[1])))
			cv2.putText(img, predict_str, pos, FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	fps_str = 'fps: %d'%(1/(time.time() - start))
	cv2.putText(img, fps_str, (10, 50), FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)
	return img
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--select_largest', type=bool,
		help='Select the largest face in image/video',
		default=True)

	subparsers = parser.add_subparsers(dest='MODE',
		help='IMAGE/VIDEO/WEBCAM')

	image_parser = subparsers.add_parser('IMAGE',
		help='Predict people in a image')

	image_parser.add_argument('--image_path', type=str,
		help='Image path',
		default='../Dataset/Test/3.jpg')


	video_parser = subparsers.add_parser('VIDEO',
		help='Predict people in a video')

	video_parser.add_argument('--video_path', type=str,
		help='Video path',
		default='../../1.mp4')


	webcam_parser = subparsers.add_parser('WEBCAM',
		help='Predict people thought a webcam')

	args = parser.parse_args()

	if args.MODE == 'IMAGE':
		ImageMain(args)
	if args.MODE == 'VIDEO' or args.MODE == 'WEBCAM':
		VideoMain(args)