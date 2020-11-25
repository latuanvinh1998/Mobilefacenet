import torch
from torch.nn import Conv2d, PReLU, MaxPool2d, Linear, Softmax
from torch.nn import Module
import numpy as np 
import os
import detect_face
import cv2

class PNet(Module):
	def __init__(self, pretrained=True):
		super().__init__()
		self.conv1 = Conv2d(3, 10, kernel_size=3)
		self.prelu1 = PReLU(10)
		self.pool1 = MaxPool2d(2, 2, ceil_mode=True)
		self.conv2 = Conv2d(10, 16, kernel_size=3)
		self.prelu2 = PReLU(16)
		self.conv3 = Conv2d(16, 32, kernel_size=3)
		self.prelu3 = PReLU(32)
		self.conv4_1 = Conv2d(32, 2, kernel_size=1)
		self.softmax4_1 = Softmax(dim=1)
		self.conv4_2 = Conv2d(32, 4, kernel_size=1)
		self.training = False
		if pretrained:
			state_dict_path = os.path.join(os.path.dirname(__file__), 'pnet.pt')
			state_dict = torch.load(state_dict_path)
			self.load_state_dict(state_dict)

	def forward(self, x):
		x = self.conv1(x)
		x = self.prelu1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.prelu2(x)
		x = self.conv3(x)
		x = self.prelu3(x)
		a = self.conv4_1(x)
		a = self.softmax4_1(a)
		b = self.conv4_2(x)
		return b, a

class RNet(Module):
	def __init__(self, pretrained=True):
		super().__init__()
		self.conv1 = Conv2d(3, 28, kernel_size=3)
		self.prelu1 = PReLU(28)
		self.pool1 = MaxPool2d(3, 2, ceil_mode=True)
		self.conv2 = Conv2d(28, 48, kernel_size=3)
		self.prelu2 = PReLU(48)
		self.pool2 = MaxPool2d(3, 2, ceil_mode=True)
		self.conv3 = Conv2d(48, 64, kernel_size=2)
		self.prelu3 = PReLU(64)
		self.dense4 = Linear(576, 128)
		self.prelu4 = PReLU(128)
		self.dense5_1 = Linear(128, 2)
		self.softmax5_1 = Softmax(dim=1)
		self.dense5_2 = Linear(128, 4)
		self.training = False 
		if pretrained:
			state_dict_path = os.path.join(os.path.dirname(__file__), 'rnet.pt')
			state_dict = torch.load(state_dict_path)
			self.load_state_dict(state_dict)

	def forward(self, x):
		x = self.conv1(x)
		x = self.prelu1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.prelu2(x)
		x = self.pool2(x)
		x = self.conv3(x)
		x = self.prelu3(x)
		x = x.permute(0, 3, 2, 1).contiguous()
		x = self.dense4(x.view(x.shape[0], -1))
		x = self.prelu4(x)
		a = self.dense5_1(x)
		a = self.softmax5_1(a)
		b = self.dense5_2(x)
		return b, a

class ONet(Module):
	def __init__(self, pretrained=True):
		super().__init__()
		self.conv1 = Conv2d(3, 32, kernel_size=3)
		self.prelu1 = PReLU(32)
		self.pool1 = MaxPool2d(3, 2, ceil_mode=True)
		self.conv2 = Conv2d(32, 64, kernel_size=3)
		self.prelu2 = PReLU(64)
		self.pool2 = MaxPool2d(3, 2, ceil_mode=True)
		self.conv3 = Conv2d(64, 64, kernel_size=3)
		self.prelu3 = PReLU(64)
		self.pool3 = MaxPool2d(2, 2, ceil_mode=True)
		self.conv4 = Conv2d(64, 128, kernel_size=2)
		self.prelu4 = PReLU(128)
		self.dense5 = Linear(1152, 256)
		self.prelu5 = PReLU(256)
		self.dense6_1 = Linear(256, 2)
		self.softmax6_1 = Softmax(dim=1)
		self.dense6_2 = Linear(256, 4)
		self.dense6_3 = Linear(256, 10)

		self.training = False 
		if pretrained:
			state_dict_path = os.path.join(os.path.dirname(__file__), 'onet.pt')
			state_dict = torch.load(state_dict_path)
			self.load_state_dict(state_dict)

	def forward(self, x):
		x = self.conv1(x)
		x = self.prelu1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.prelu2(x)
		x = self.pool2(x)
		x = self.conv3(x)
		x = self.prelu3(x)
		x = self.pool3(x)
		x = self.conv4(x)
		x = self.prelu4(x)
		x = x.permute(0, 3, 2, 1).contiguous()
		x = self.dense5(x.view(x.shape[0], -1))
		x = self.prelu5(x)
		a = self.dense6_1(x)
		a = self.softmax6_1(a)
		b = self.dense6_2(x)
		c = self.dense6_3(x)
		return b, c, a

class MTCNN(Module):
	def __init__(self, image_size=160, device=None):
		super().__init__()
		self.image_size = image_size

		self.pnet = PNet()
		self.rnet = RNet()
		self.onet = ONet()

		self.device = torch.device('cpu')
		if device is not None:
			self.device = device
			self.to(device)

	def detect(self, img, select_largest=True, proba=False ,landmarks=False):
		with torch.no_grad():
			batch_boxes, batch_points = detect_face.detect_face(img, self.pnet, self.rnet, self.onet, self.device)
		boxes, probs, points = [], [], []
		for box, point in zip(batch_boxes, batch_points):
			box = np.array(box)
			point = np.array(point)
			if len(box) == 0:
				if landmarks and proba:
					return None, None, None
				elif landmarks and not proba:
					return None, None
				elif not landmarks and proba:
					return None, None
				return None
			elif select_largest:
				box_order = np.argsort((box[:,2]-box[:,0])*(box[:,3]-box[:,1]))[::-1]
				box = box[box_order][[0]]
				point = point[box_order][[0]]
				boxes.append(box[:, :4])
				probs.append(box[:, 4])
				points.append(point)
			else:
				boxes.append(box[:, :4])
				probs.append(box[:, 4])
				points.append(point)

		boxes = np.float32(boxes)
		probs = np.float32(probs)
		points = np.float32(points)
		
		boxes = boxes[0]
		points = points[0]

		if landmarks and proba:
			return boxes, probs, points
		elif landmarks and not proba:
			return boxes, points
		elif not landmarks and proba:
			return boxes, probs
		return boxes

	def align(self, img, select_largest=True, save_path=None):
		assert img is not None
		thumbnails = []
		boxes, points = self.detect(img, select_largest=select_largest, proba=False, landmarks=True)
		if boxes is not None:
			i = 0
			for box, point in zip(boxes, points):
				i+=1
				thumbnail = detect_face.extract_face(img, box, self.image_size)
				thumbnails.append(thumbnail)
				if save_path is not None:
					save_img(thumbnail, save_path+'face_'+str(i)+'.jpg')
		else:
			return None
		return thumbnails

def save_img(img, save_path):
	os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
	if isinstance(img, np.ndarray):
		cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
	else:
		img.save(save_path)
	print("Image Saved: "+ os.path.expanduser(save_path))