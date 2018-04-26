from model.MTCNN import MTCNN
import cv2
import argparse
from pathlib import Path
from PIL import Image
from model.MTCNN import MTCNN
from datetime import datetime
from multiprocessing import Process, Pipe,Value,Array
import time
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import pdb
from PIL import Image
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from model.MTCNN import MTCNN
from model.Cosface import Cosface
from torchvision import transforms as trans
from utils import *

conf = EasyDict()
conf.use_cuda = torch.cuda.is_available()
conf.pin_mem = conf.use_cuda  # the is no camera connected to my ubuntu machine, so it's haven't been tested under cuda environment
conf.input_size = (128,92)
conf.num_classes = 51332 # training parameter, useless in here
conf.board_loss_every = 500 # training parameter, useless in here
conf.eva_every = 5000 # training parameter, useless in here
conf.save_every = 30000 # training parameter, useless in here
conf.model_path = 'model/cosface_best_cpu.pth'


root = Path.cwd()
mtcnn = MTCNN() # adapted MTCNN

def crop_face(img):
    box = mtcnn.detect_faces(img,min_face_size=120)[0][:-1].astype(int)
    # min_face_size is the MTCNN parameter, smaller this value, slower the computation, but you can detect smaller faces
    box + np.array([-1,-1,1,1]) # personal choice
    return crop_box(img,box)
transform_cropping = trans.Compose([
    trans.Lambda(lambda x : crop_face(x)),
    trans.Resize(conf.input_size),
    trans.ToTensor(),
    trans.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
transform_normal = trans.Compose([
    trans.Resize(conf.input_size),
    trans.ToTensor(),
    trans.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

face_model = Cosface(conf)

face_model.load_state(root/conf.model_path)

cropped_face_folder = Path('C:\\Users\\l00271155\\face\\AM_Softmax\\facebook\\cropped')

original_face_folder = Path('C:\\Users\\l00271155\\face\\AM_Softmax\\facebook\\original')

names_cropped,features_cropped = prepare_faceid(cropped_face_folder,face_model,transform_normal,conf)

names_orig,features_orig = prepare_faceid(original_face_folder,face_model,transform_cropping,conf)

names = np.concatenate((names_cropped,names_orig,np.array(['unknown'])))

features = np.concatenate((features_cropped,features_orig))

"""
load the face features and names from cropped_folder(previously taken by camera) and original_folder(uncropped)
"""

def face_reader(conn,flag,boxes_arr,result_arr,threshold):
    while True:
        try:
            image = conn.recv()
        except:
            continue
        try:
#             
            bboxes = mtcnn.detect_faces(image)
        except:
#             image.save('abnormal.jpg')
            bboxes = []
        if len(bboxes) > 0:
            print('bboxes in reader : {}'.format(bboxes))
            bboxes = bboxes[:10,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice
            faces = []
            for i in range(len(bboxes)):
                img = image.crop(bboxes[i])
                img = transform_normal(img)
                img = Variable(img).unsqueeze(0)
                face = face_model(img).data.numpy()
                faces.append(face)
            faces = np.concatenate(faces,0) #shape:[detected face number,512]
            results = lookup_face(faces,features,names,threshold)
            assert bboxes.shape[0] == results.shape[0],'bbox and faces number not same'
            bboxes = bboxes.reshape([-1])
            for i in range(len(boxes_arr)):
                if i < len(bboxes):
                    boxes_arr[i] = bboxes[i]
                else:
                    boxes_arr[i] = 0 
            for i in range(len(result_arr)):
                if i < len(results):
                    result_arr[i] = results[i]
                else:
                    result_arr[i] = -1 
        else:
            for i in range(len(boxes_arr)):
                boxes_arr[i] = 0 # by default,it's all 0
            for i in range(len(result_arr)):
                result_arr[i] = -1 # by default,it's all -1
        print('boxes_arr ： {}'.format(boxes_arr[:4]))
        print('result_arr ： {}'.format(result_arr[:4]))
        flag.value = 0