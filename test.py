from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from align_dataset import normal_align
from create_svm import create_svm

import tkinter
import cv2
import PIL.Image, PIL.ImageTk

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
from mtcnn import MTCNN, detect_face

from threading import Thread


window = Tk()
window.title('Face Recognition Terminal')

photo = None
switch = 0

FONT = cv2.FONT_HERSHEY_SIMPLEX 

MAX_IMGS = 100
NROF_IMGS = 0

SVM_MODEL =  "../PretrainedModel/classifier.pkl"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(image_size=112, device=device)

model = MobileFaceNet(512).to(torch.device("cuda:0"))
model.load_state_dict(torch.load('../PretrainedModel/model.pth'))

with open(SVM_MODEL, 'rb') as infile:
	(class_name, svm_model) = pickle.load(infile)
	
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

cap = cv2.VideoCapture(0)

def handleswitch():

	global switch
	switch = 1 - switch

def handleaddid():

	global switch
	global class_name, svm_model
	
	switch = 0
	SVM_MODEL =  "../PretrainedModel/classifier.pkl"


	if text_input.get() != '':

		add(text_input.get())
		normal_align()
		create_svm()

		with open(SVM_MODEL, 'rb') as infile:
			(class_name, svm_model) = pickle.load(infile)
		print('load done')

		return

	else:
		label.configure(text='Must Add Name of Identify')
		return

def save_face(img, save_dir):
	global NROF_IMGS, detector
	boxes, proba, points = detector.detect(img, select_largest=True, landmarks=True, proba=True)
	if boxes is not None:
		for box, point in zip(boxes, points):
			if NROF_IMGS < MAX_IMGS and proba > 0.99:
				NROF_IMGS+=1
				face = detect_face.extract_face(img, box, image_size=160)
				os.makedirs(os.path.dirname(save_dir + "%d.jpg"%(NROF_IMGS)) + "/", exist_ok=True)
				cv2.imwrite(save_dir + "%d.jpg"%(NROF_IMGS), face)
				print("Image Saved: "+ os.path.expanduser(save_dir + "%d.jpg"%(NROF_IMGS)))
	return

def add(name):
	nrof_lines = 256
	SAVE_DIR = "../Dataset/Processed/" + name + '/'

	if not cap.isOpened:
		raise RuntimeError("Cannot Open Video")
	
	add = 0

	start = time.time()
	while(add == 0):
		ret, frame = cap.read()	
		img = cv2.flip(frame, 1)
		if frame is None:
			break
		(h, w) = img.shape[:2]
		radius = min(h, w)/2 - 50
		cv2.circle(img, (w//2, h//2), min(h,w), (0, 0, 0), min(h,w)+150)
		for i in range(np.int32(nrof_lines)):
			angle = 2 * math.pi * i / nrof_lines
			x, y = w//2 + math.sin(angle)*radius, h//2 + math.cos(angle)*radius
			pt1 = x, y
			pt2 = x + math.sin(angle)*10, y + math.cos(angle)*10
			cv2.line(img, tuple(np.int32(pt1)), tuple(np.int32(pt2)), (255, 255, 255), 1)

		percent = NROF_IMGS/MAX_IMGS
		nrof_big_lines = nrof_lines * percent
		n = 0
		while n < nrof_big_lines :
			angle = 2 * math.pi * n / nrof_lines
			x, y = w//2 + math.sin(angle)*radius, h//2 + math.cos(angle)*radius
			pt1 = x, y
			pt2 = x + math.sin(angle)*15, y + math.cos(angle)*15
			cv2.line(img, tuple(np.int32(pt1)), tuple(np.int32(pt2)), (255, 255, 255), 2)
			n += 1
		
		thread = Thread(target=save_face, args=(img, SAVE_DIR,))
		thread.start()
		start = time.time()
		
		boxes = detector.detect(img, select_largest=True, landmarks=False, proba=False)
		if boxes is not None:
			for box in boxes:
				cv2.putText(img, "Turn Head Around!", (10, 50), FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)
				cv2.rectangle(img, tuple((np.int32(box[0]), np.int32(box[1]))), tuple((np.int32(box[2]), np.int32(box[3]))), (255, 255, 255), 1)
		if NROF_IMGS == MAX_IMGS:
			done_logo = cv2.imread("../Dataset/done.jpg")
			done_logo = cv2.resize(done_logo, (200, 200))
			done_logo = cv2.addWeighted(img[h//2-100:h//2+100,w//2-100:w//2+100,:],0.1,done_logo[0:200,0:200,:],0.9,0)
			img[h//2-100:h//2+100,w//2-100:w//2+100,:] = done_logo
			cv2.putText(img, "Press ""Q"" to Exit.", (h//2-100, w//2-100), FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.imshow('preview', np.uint8(img))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# cap.release()
	cv2.destroyAllWindows()

def Recognition(img):
	start = time.time()
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

def Processing():

	global canvas, photo, switch

	if switch == 1:

		ret, frame = cap.read()
	    
		frame = Recognition(frame)
		frame = cv2.resize(frame, (800,450))

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))

		canvas.create_image(0,0, image = photo, anchor=tkinter.NW)


	elif switch == 0:

		frame = cv2.imread('nope.jpg')
		frame = cv2.resize(frame, (800,450))
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))

		canvas.create_image(0,0, image = photo, anchor=tkinter.NW)

	window.after(15, Processing)


canvas = Canvas(window, width = 800, height= 450 , bg= "gray")
canvas.pack()

label = tkinter.Label(window, text='Add Id', font=('Times New Roman', 20))
label.pack()

text_input = Entry(window, width=10)
text_input.pack()

function = Button(window,text = "Add one more Identify", command=handleaddid)
function.pack()

start = Button(window,text = "Start/Stop Get Frame", command=handleswitch)
start.pack()

Processing()


window.mainloop()