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
from face_reader import face_reader
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-t','--threshold',help='threshold to decide identical faces',default=0.36, type=float)
    args = parser.parse_args()
    root = Path.cwd()
#     save_path = 'saved_video/'

    conf = EasyDict()
    conf.use_cuda = torch.cuda.is_available()
    conf.pin_mem = conf.use_cuda
    conf.input_size = (128,92)
    conf.num_classes = 51332
    conf.board_loss_every = 500
    conf.eva_every = 5000
    conf.save_every = 30000
    conf.model_path = 'model/cosface_best_cpu.pth'

    mtcnn = MTCNN()
    def crop_face(img): # crop the face out from captured image
        box = mtcnn.detect_faces(img,min_face_size=40)[0][:-1].astype(int)
        box + np.array([-1,-1,1,1])
        return crop_box(img,box)
    transform_cropping = trans.Compose([
        trans.Lambda(lambda x : crop_face(x)),
        trans.Resize(conf.input_size),
        trans.ToTensor(),
        trans.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    # crop and resize image to input.size
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
    # prepare the saved names and features, in 2 folders, one for catured, one for downloaded, concantenate together

    # inital camera
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    if args.save:
        video_writer = cv2.VideoWriter('recording.avi',cv2.VideoWriter_fourcc(*'XVID'), 6, (1280,720))
        # frame rate 6 due to my laptop is quite slow...
    parent_conn, child_conn = Pipe()
    """
    using pipe to send image from main process to child process
    use multi-process sharing methods to save(in child) and load(in main) detected boxes and names    
    """
    boxes = Array('i',range(40)) # all detected face bboxes, at maximum 10 faces
    result = Array('i',range(10)) # all detected names,at maximum 10 faces
    flag = Value('d', 0.0) # whether updated the boxes and faces
    p = Process(target=face_reader, args=(child_conn,flag,boxes,result,args.threshold))
    p.start()
    while cap.isOpened():
        isSuccess,frame = cap.read()
        if isSuccess:
            if flag.value == 0:
                image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                parent_conn.send(image)
                flag.value == 1
            box_list = []
            display_names = []
            print('boxes_arr_main ： {}'.format(boxes))
            print('result_arr_main ： {}'.format(result))
            # if there are detected faces, former 2 lines will print result
            if np.sum(boxes) > 0: # if there is no face detected, numbers in boxes shuld all be 0
                for i in range(10):
                    if boxes[i*4] == 0:
                        break
                        # box saved format : [b1x1,b1y1,b1x2,b1y2,b2x1,b2y1,b2x2,b2y2,.....,0,0,0,0,0....]
                        # if encounter 0, means there is no more boxes
                    box_list.append([boxes[i*4],boxes[i*4+1],boxes[i*4+2],boxes[i*4+3]])
                    """ 
                    there is no high level data format in multiprocess, only array,
                    so I have to save the four number of each box in a row, and at maximum 10*4=40 numbers
                    """
                    display_names.append(names[result[i]])
                    
                for idx,bbox in enumerate(box_list):
                    frame = draw_box_name(bbox,display_names[idx],frame)
    
        cv2.imshow("My Capture",frame)
        if args.save:
            video_writer.write(frame)

        if cv2.waitKey(1)&0xFF == ord('q'):
            p.terminate()
            break

    cap.release()
    if args.save:
        video_writer.release()
    cv2.destroyAllWindows()