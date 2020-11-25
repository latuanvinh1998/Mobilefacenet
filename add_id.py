import cv2
import time
import numpy as np
import math
import sys
import torch
import os
import argparse
from threading import Thread
sys.path.insert(1, "MTCNN")
from mtcnn import MTCNN, detect_face
MAX_IMGS = 100
NROF_IMGS = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(device=device)

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

	cap = cv2.VideoCapture('http://192.168.1.5:4747/video')
	if not cap.isOpened:
		raise RuntimeError("Cannot Open Video")
	
	start = time.time()
	while(True):
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

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('name', type=str,
		help='Name of identify')

	args = parser.parse_args()
	main(args)