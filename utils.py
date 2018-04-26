from torchvision.utils import make_grid
import numpy as np
from torch.autograd import Variable
import torch
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
import pdb
from PIL import Image
import cv2
import numpy as np
import PIL

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm.expand_as(input))
    return output

def lookup_face(faces,features,names,threshold):
    verify_mat = np.dot(faces,features.T)
    most_alike_idx = np.argmax(verify_mat,1)
    decition = verify_mat[range(verify_mat.shape[0]),most_alike_idx] > threshold
    lookup_result = np.where(decition,most_alike_idx,-1)
    return lookup_result

def lookup_name(faces,features,names,threshold):
    verify_mat = np.dot(faces,features.T)
    most_alike_idx = np.argmax(verify_mat,1)
    most_alike_names = names[most_alike_idx]
    decition = verify_mat[range(verify_mat.shape[0]),most_alike_idx] > threshold
    lookup_result = np.where(decition,most_alike_names,'unknow')
    return lookup_result

def draw_box_name(bbox,name,frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2,
                    (255,255,255),
                    3,
                    cv2.LINE_AA)
    return frame

def de_preprocess(x):
    return x*0.5 + 0.5

def crop_box(img,box):
    if isinstance(img,PIL.JpegImagePlugin.JpegImageFile):
        img = img.crop(box)
    elif isinstance(img,np.ndarray):
        img = img[box[1]:box[3],box[0]:box[2]]
    return img

def get_folder_files(path):
    folders = []
    get_files(path,folders)
    return folders

def get_files(path,folders):
    for p in path.iterdir():
        if p.is_dir():
            get_files(p,folders)
        elif p.is_file():
            folders.append(p)
    return folders

# def get_faceid(root,model,transform,conf):
#     pathes = get_folder_files(root)
#     features = []
#     for path in pathes:
#         img = Image.open(path)
#         img = transform(img).unsqueeze(0)
#         img = Variable(img)
#         if conf.use_cuda:
#             img = img.cuda()
#         features.append(model(img).squeeze(0).data.numpy())
#     mean_feature = np.array(features).mean(axis=0)
#     mean_feature = normalize(mean_feature)
#     return root.name,mean_feature

def get_faceid(root,model,transform,conf):
    pathes = get_folder_files(root)
#     import pdb
#     pdb.set_trace()
    assert len(pathes) == 1,'contrains more than 1 image in the folder!'
    features = []
    path = pathes[0]
    img = Image.open(path)
    img = transform(img).unsqueeze(0)
    img = Variable(img)
    if conf.use_cuda:
        img = img.cuda()
    feature = model(img).squeeze(0).data.numpy()
    return root.name,feature

def prepare_faceid(root,model,transform,conf):
    names = []
    features = []
    for path in root.iterdir():
        name,feature = get_faceid(path,model,transform,conf)
        if name not in names:
            names.append(name)
            features.append(feature)
    return np.array(names,str),np.array(features)

def l2_distance(x1,x2):
    assert x1.size() == x2.size(),'x1,size : {},x2.size : {}'.format(x1.size(),x2.size())
    eps = 1e-4 / x1.size(1)
    diff = torch.abs(x1 - x2)
    out = torch.pow(diff, 2).sum(dim=1)
    return torch.pow(out + eps, 1. / 2)

def show_lfw_errors(model,lfw_loader,conf,total=16,threshold=1.127):
    model.eval()
    num = 0
    if conf.use_cuda:
        model.cuda()
    pos1_error_list = []
    pos2_error_list = []
    pos_neg_error_list = []
    neg_error_list = []
    rounds = total//lfw_loader.batch_size + 1
    pos_incorrect_num = 0
    neg_incorrect_num = 0
    while pos_incorrect_num <total or neg_incorrect_num<total:
        try:
            pos1,pos2,neg,_,_ = next(iter(lfw_loader))
        except:
            continue
        num += pos1.shape[0]
        pos1 = Variable(pos1,volatile=True)
        pos2 = Variable(pos2,volatile=True)
        neg = Variable(neg,volatile=True)
        if conf.use_cuda:
            pos1,pos2,neg = pos1.cuda(),pos2.cuda(),neg.cuda()
        pos1_fea = model.forward(pos1)
        pos2_fea = model.forward(pos2)
        if pos_incorrect_num < total:
            pos_dis = l2_distance(pos1_fea,pos2_fea).data.cpu().numpy()
            pos_errors_ind = np.where(pos_dis > threshold)[0].tolist()
            if len(pos_errors_ind) > 0:
                pos1_incrorrect = pos1.cpu().data[pos_errors_ind]
                pos2_incrorrect = pos2.cpu().data[pos_errors_ind]
                pos1_error_list.append(pos1_incrorrect)
                pos2_error_list.append(pos2_incrorrect)
                pos_incorrect_num += pos1_incrorrect.shape[0]
                print('pos num : {}'.format(pos_incorrect_num))
        if neg_incorrect_num < total:
            neg_fea = model.forward(neg)
            neg1_dis = l2_distance(pos1_fea,neg_fea).data.cpu().numpy()
            neg2_dis = l2_distance(pos2_fea,neg_fea).data.cpu().numpy()
            neg1_errors_ind = np.where(neg1_dis < threshold)[0].tolist()
            if len(neg1_errors_ind) > 0:
                neg1_incorrect = neg.cpu().data[neg1_errors_ind]
                pos1_neg_incorrect = pos1.cpu().data[neg1_errors_ind]
                neg_incorrect_num += neg1_incorrect.shape[0]
                pos_neg_error_list.append(pos1_neg_incorrect)
                neg_error_list.append(neg1_incorrect)
            neg2_errors_ind = np.where(neg2_dis < threshold)[0].tolist()
            if len(neg2_errors_ind) > 0:
                neg2_incorrect = neg.cpu().data[neg2_errors_ind]
                pos2_neg_incorrect = pos2.cpu().data[neg2_errors_ind]
                neg_incorrect_num += neg2_incorrect.shape[0]
                print('neg num : {}'.format(neg_incorrect_num))
                pos_neg_error_list.append(pos2_neg_incorrect)
                neg_error_list.append(neg2_incorrect)
    pos1_errors = torch.cat(pos1_error_list)[:total]
    pos2_errors = torch.cat(pos2_error_list)[:total]
    pos_neg_errors = torch.cat(pos_neg_error_list)[:total]
    neg_errors = torch.cat(neg_error_list)[:total]
    grid = torch.cat([pos1_errors,pos2_errors,pos_neg_errors,neg_errors])
    return trans.ToPILImage()(de_preprocess(make_grid(grid,nrow=16)))
