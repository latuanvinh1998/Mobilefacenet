import argparse
import cv2
import os
import torch
import numpy as np 
import sys
sys.path.insert(1, "MTCNN")
from mtcnn import MTCNN
sys.path.insert(1, "Facenet")
from load_dataset import load_dataset


def alignMain(args):
	LOG_DIR = '../Dataset/LogError/'

	os.makedirs(os.path.dirname(LOG_DIR) + "/", exist_ok=True)
	f = open(LOG_DIR+'AlignErrortxt', 'w')

	if not os.path.exists(args.rawdataDir):
		raise UserWarning("Input Dataset Directory does not exist!")
	os.makedirs(os.path.dirname(args.outputDir) + "/", exist_ok=True)

	dataset = load_dataset(args.rawdataDir)
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	detector = MTCNN(image_size=args.size, device=device)

	for i in range(len(dataset)):
		print("Working in folder " + dataset[i].name +" with index " + str(i), end='')
		for path in dataset[i].paths:
			_, file_name = os.path.split(path)
			imgBGR = cv2.imread(path)
			thumbnails = detector.align(img=imgBGR, select_largest=True)
			if thumbnails is not None:
				thumbnail = thumbnails[-1]
				# thumbnail = cv2.cvtColor(np.float32(thumbnail), cv2.COLOR_RGB2BGR)
				OUTPUT_DIR = args.outputDir + dataset[i].name + "/"
				os.makedirs(os.path.dirname(OUTPUT_DIR) + "/", exist_ok=True)
				str1 = "{}{}".format(OUTPUT_DIR, file_name)
				cv2.imwrite(str1, thumbnail)
			else:
				f.write("Unable to detect faces: {}/{}. \n".format(dataset[i].name, file_name))
		print(' ... Done!')
	f.close()

def normal_align():
	LOG_DIR = '../Dataset/LogError/'

	os.makedirs(os.path.dirname(LOG_DIR) + "/", exist_ok=True)
	f = open(LOG_DIR+'AlignErrortxt', 'w')

	if not os.path.exists('../Dataset/Raw/'):
		raise UserWarning("Input Dataset Directory does not exist!")
	os.makedirs(os.path.dirname('../Dataset/Processed/') + "/", exist_ok=True)

	dataset = load_dataset('../Dataset/Raw/')
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	detector = MTCNN(image_size=112, device=device)

	for i in range(len(dataset)):
		for path in dataset[i].paths:
			_, file_name = os.path.split(path)
			imgBGR = cv2.imread(path)
			thumbnails = detector.align(img=imgBGR, select_largest=True)
			if thumbnails is not None:
				thumbnail = thumbnails[-1]
				# thumbnail = cv2.cvtColor(np.float32(thumbnail), cv2.COLOR_RGB2BGR)
				OUTPUT_DIR = '../Dataset/Processed/' + dataset[i].name + "/"
				os.makedirs(os.path.dirname(OUTPUT_DIR) + "/", exist_ok=True)
				str1 = "{}{}".format(OUTPUT_DIR, file_name)
				cv2.imwrite(str1, thumbnail)
			else:
				f.write("Unable to detect faces: {}/{}. \n".format(dataset[i].name, file_name))
	f.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--rawdataDir', type=str,
		help='Raw Data Directory',
		default='../Dataset/Raw/')

	parser.add_argument('--outputDir', type=str,
		help='Aligned image saving directory',
		default='../Dataset/Processed/')

	parser.add_argument('--size', type = int, 
		help="Precessed images dimension",
		default=112)

	parser.add_argument('-m', '--multiFace', type=bool,
		help='Allow align Multi-faces',
		default=False)

	args = parser.parse_args()
	alignMain(args)

