from torchvision.ops.boxes import batched_nms
from torch.nn.functional import interpolate
import numpy as np 
import torch
import cv2
import os 

def detect_face(imgs, pnet, rnet, onet, device):
	MINSIZE = 20
	FACTOR = 0.709
	THRESHOLD = [0.6, 0.7, 0.7]

	if imgs is None:
		raise ValueError("Image is not found")
	if isinstance(imgs, (np.ndarray, torch.Tensor)):
		if isinstance(imgs, np.ndarray):
			imgs = torch.as_tensor(imgs.copy(), device=device)
		if isinstance(imgs, torch.Tensor):
			imgs = torch.as_tensor(imgs, device=device)
		if len(imgs.shape) == 3:
			imgs = imgs.unsqueeze(0)
	else:
		if not isinstance(imgs, (list, tuple)):
			imgs = [imgs]
		if any(img.size != imgs[0].size for img in imgs):
			raise Exception("MTCNN batch processing only compatible with equal-dimension images.")
		imgs = np.stack([np.uint8(img) for img in imgs])
		imgs = torch.as_tensor(imgs.copy(), device=device)
	
	model_dtype = next(pnet.parameters()).dtype
	imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

	img_height, img_width = imgs.shape[2:4]
	min_face_size = min(img_height, img_width) * (12.0/MINSIZE)

	#Create scale pyramid
	scale_i = 12.0/MINSIZE
	scales = []
	while min_face_size >= 12:
		scales.append(scale_i)
		scale_i = scale_i * FACTOR
		min_face_size = min_face_size * FACTOR

	#First stage
	boxes = []
	image_idxs = []
	scale_picks = []

	offset = 0
	for scale in scales:
		im_data = interpolate(imgs, (int(img_height*scale+1), int(img_width*scale+1)), mode="area")
		im_data = (im_data - 127.5) * 0.0078125
		reg, probs = pnet(im_data)

		boxes_scale, image_idxs_scale = generateBoundingBox(reg, probs[:, 1], scale, THRESHOLD[0])
		boxes.append(boxes_scale)
		image_idxs.append(image_idxs_scale)

		pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_idxs_scale, 0.5)
		scale_picks.append(pick + offset)
		offset += boxes_scale.shape[0]

	boxes = torch.cat(boxes, dim=0)
	image_idxs = torch.cat(image_idxs, dim=0)
	scale_picks = torch.cat(scale_picks, dim=0)

	#NMS scale + image
	boxes, image_idxs = boxes[scale_picks], image_idxs[scale_picks]
	#NMS image
	pick = batched_nms(boxes[:, :4], boxes[:, 4], image_idxs, 0.7)
	boxes, image_idxs = boxes[pick], image_idxs[pick]

	regw = boxes[:, 2] - boxes[:, 0]
	regh = boxes[:, 3] - boxes[:, 1]
	qq1 = boxes[:, 0] + boxes[:, 5] * regw
	qq2 = boxes[:, 1] + boxes[:, 6] * regh
	qq3 = boxes[:, 2] + boxes[:, 7] * regw
	qq4 = boxes[:, 3] + boxes[:, 8] * regh
	boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
	boxes = rerec(boxes)

	#Second Stage
	if len(boxes) > 0:
		y, ey, x, ex = pad(boxes, img_width, img_height)
		im_data = []
		for k in range(len(y)):
			if ey[k] > (y[k]-1) and ex[k] > (x[k]-1):
				img_k = imgs[image_idxs[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
				im_data.append(interpolate(img_k, (24, 24), mode="area"))
		im_data = torch.cat(im_data, dim=0)
		im_data = (im_data - 127.5) * 0.0078125

		out = fixed_batch_process(im_data, rnet)
		out0 = out[0].permute(1, 0)
		out1 = out[1].permute(1, 0)
		score = out1[1, :]
		ipass = score > THRESHOLD[1]
		boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
		image_idxs = image_idxs[ipass]
		mv = out0[:, ipass].permute(1, 0)

		#NMS image
		pick = batched_nms(boxes[:, :4], boxes[:, 4], image_idxs, 0.7)
		boxes, image_idxs, mv = boxes[pick], image_idxs[pick], mv[pick]
		boxes = bbreg(boxes, mv)
		boxes = rerec(boxes)

	#Third Stage
	points = torch.zeros(0, 5, 2, device=device)
	if len(boxes) > 0:
		y, ey, x, ex = pad(boxes, img_width, img_height)
		im_data = []
		for k in range(len(y)):
			if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
				img_k = imgs[image_idxs[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
				im_data.append(interpolate(img_k, (48, 48), mode="area"))
		im_data = torch.cat(im_data, dim=0)
		im_data = (im_data - 127.5) * 0.0078125

		out = fixed_batch_process(im_data, onet)
		out0 = out[0].permute(1, 0)
		out1 = out[1].permute(1, 0)
		out2 = out[2].permute(1, 0)
		score = out2[1, :]
		points = out1
		ipass = score > THRESHOLD[2]
		points = points[:, ipass]
		boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
		image_idxs = image_idxs[ipass]
		mv = out0[:, ipass].permute(1, 0)

		w_i = boxes[:, 2] - boxes[:, 0] + 1
		h_i = boxes[:, 3] - boxes[:, 1] + 1
		points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
		points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
		points = torch.stack((points_x, points_y)).permute(2, 1, 0)
		boxes = bbreg(boxes, mv)

		pick = batched_nms(boxes[:, :4], boxes[:, 4], image_idxs, 0.7)
		boxes, image_idxs, points = boxes[pick], image_idxs[pick], points[pick]

	boxes = boxes.cpu().numpy()
	points = points.cpu().numpy()
	image_idxs = image_idxs.cpu().numpy()
	batch_boxes = []
	batch_points = []

	batch_size = len(imgs)
	for b_i in range(batch_size):
		b_i_inds = np.where(image_idxs == b_i)
		batch_boxes.append(boxes[b_i_inds].copy())
		batch_points.append(points[b_i_inds].copy())
	batch_boxes, batch_points = np.array(batch_boxes), np.array(batch_points)
	return batch_boxes, batch_points

def fixed_batch_process(im_data, model):
	batch_size = 512
	out = []
	for i in range(0, len(im_data), batch_size):
		batch = im_data[i:(i+batch_size)]
		out.append(model(batch))

	return tuple(torch.cat(v, dim=0) for v in zip(*out))


def generateBoundingBox(reg, probs, scale, thresh):
	stride = 2 
	cellsize = 12
	reg = reg.permute(1, 0, 2, 3)
	
	mask = probs >= thresh
	mask_idxs = mask.nonzero(as_tuple=False)
	image_idxs = mask_idxs[:, 0]
	
	score = probs[mask]
	reg = reg[:, mask].permute(1, 0)
	bb = mask_idxs[:, 1:].type(reg.dtype).flip(1)
	q1 = ((stride*bb+1)/scale).floor()
	q2 = ((stride*bb+cellsize)/scale).floor()
	boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
	return boundingbox, image_idxs


def rerec(bb):
	h = bb[:, 3] - bb[:, 1]
	w = bb[:, 2] - bb[:, 0]
	l = torch.max(w, h)
	bb[:, 0] = bb[:, 0] + w * 0.5 - l * 0.5
	bb[:, 1] = bb[:, 1] + h * 0.5 - l * 0.5
	bb[:, 2:4] = bb[:, :2] + l.repeat(2, 1).permute(1, 0)
	return bb


def pad(boxes, w, h):
	boxes = boxes.trunc().int().cpu().numpy()
	x = boxes[:, 0]
	y = boxes[:, 1]
	ex = boxes[:, 2]
	ey = boxes[:, 3]
	x[x < 1] = 1
	y[y < 1] = 1
	ex[ex > w] = w
	ey[ey > h] = h
	return y, ey, x, ex

def bbreg(boundingbox, reg):
	if reg.shape[1] == 1:
		reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

	w = boundingbox[:, 2] - boundingbox[:, 0] + 1
	h = boundingbox[:, 3] - boundingbox[:, 1] + 1
	b1 = boundingbox[:, 0] + reg[:, 0] * w
	b2 = boundingbox[:, 1] + reg[:, 1] * h
	b3 = boundingbox[:, 2] + reg[:, 2] * w
	b4 = boundingbox[:, 3] + reg[:, 3] * h
	boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)
	return boundingbox

def extract_face(img, box, image_size=160):

	if isinstance(img, (np.ndarray, torch.Tensor)):
		raw_image_size = img.shape[1::-1]
	else:
		raw_image_size = img.size
	box = [
		int(max(box[0], 0)),
		int(max(box[1], 0)),
		int(min(box[2], raw_image_size[0])),
		int(min(box[3], raw_image_size[1])),
	]

	if isinstance(img, np.ndarray):
		img = img[box[1]:box[3], box[0]:box[2]]
		face = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA).copy()
	elif isinstance(img, torch.Tensor):
		img = img[box[1]:box[3], box[0]:box[2]]
		face = imresample(img.permute(2, 0, 1).unsqueeze(0).float(),(image_size, image_size)).byte().squeeze(0).permute(1, 2, 0)
	else:
		face = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)

	return face
