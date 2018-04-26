import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms as trans
from torch import optim
from torch.autograd import Variable
from torch import nn
import numpy as np
from tqdm import tqdm
from utils import *
from datetime import datetime
from model.net import Sphere20a

class Cosface(nn.Module):
    def __init__(self,conf):
        super(Cosface, self).__init__()
        self.model = Sphere20a(conf.num_classes) 
        self.criterion = nn.CrossEntropyLoss()
        self.step = 0
        self.use_cuda = conf.use_cuda
        self.board_loss_every = conf.board_loss_every
        self.eva_every = conf.eva_every
        self.save_every = conf.save_every
        if self.use_cuda:
            self.cuda()
        
    def forward(self, x,label=None):
        x = self.model(x,label)
        return x
    
    def fit(self,train_loader,val_trip_loader,
            val_loss_loader,lfw_loader,optimizer,
            writer,epochs=1,step=None,schedulers=None,log=''):
        self.train()
        if step:
            self.step = step
        running_center_loss = 0.
        running_ce_loss = 0.
        running_loss = 0.
        total_accuracy = 0.
        if schedulers:
            scheduler_step,scheduler_plateau = schedulers
            
        for epoch in range(epochs):
            for data,label in tqdm(iter(train_loader)):
                self.step += 1
                if schedulers:
                    scheduler_step.step()
                data,label = Variable(data),Variable(label)
                if self.use_cuda:
                    data,label = data.cuda(),label.cuda()

                theta_am = self.forward(data,label)

                loss = self.criterion(theta_am,label)
 
                # compute gradient and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.data[0]

                if self.step % self.board_loss_every == 0:
                    writer.add_scalar('loss',running_loss/self.board_loss_every,self.step)
                    running_loss = 0.
                    if schedulers:
                        scheduler_step.step()
                        
                if self.step % self.eva_every == 0:
                    total_acc,pos_acc,\
                    neg_acc,pos_dis,neg_dis,pos_std,neg_std,\
                    threshold,\
                    val_ce_loss\
                    = self.evaluate(val_trip_loader,val_loss_loader,total=3000,loss_rounds=10)
                    
                    writer.add_scalar('Total_Acc',total_acc,self.step)
                    writer.add_scalar('Positive_Acc',pos_acc,self.step)
                    writer.add_scalar('Negative_Acc',neg_acc,self.step)
                    writer.add_scalar('Positive dis',pos_dis,self.step)
                    writer.add_scalar('Negative dis',neg_dis,self.step)
                    writer.add_scalar('Positive std',pos_std,self.step)
                    writer.add_scalar('Negative std',neg_std,self.step)
                    writer.add_scalar('Pos_Neg_Average_Dis_Dis',pos_dis - neg_dis,self.step)
                    writer.add_scalar('threshold',threshold,self.step)
                    writer.add_scalar('Val Cross_Entropy Loss',val_ce_loss,self.step)
                    
                    
                if self.step % self.save_every == 0:
                    self.save_state(self.model_path,optimizer,extra=log)
                    
            lfw_total_acc,lfw_pos_acc,lfw_neg_acc,lfw_pos_dis,lfw_neg_dis,\
                lfw_pos_std,lfw_neg_std,lfw_threshold = self.lfw_evaluate(lfw_loader,total=5000)
            writer.add_scalar('lfw_Total_Acc',lfw_total_acc,self.step)
            writer.add_scalar('lfw_Positive_Acc',lfw_pos_acc,self.step)
            writer.add_scalar('lfw_Negative_Acc',lfw_neg_acc,self.step)
            writer.add_scalar('lfw_Positive dis',lfw_pos_dis,self.step)
            writer.add_scalar('lfw_Negative dis',lfw_neg_dis,self.step)
            writer.add_scalar('lfw_Positive std',lfw_pos_std,self.step)
            writer.add_scalar('lfw_Negative std',lfw_neg_std,self.step)
            writer.add_scalar('lfw_Pos_Neg_Average_Dis_Dis',lfw_pos_dis - lfw_neg_dis,self.step)
            writer.add_scalar('lfw_threshold',lfw_threshold,self.step)
            if schedulers:
                scheduler_plateau.step(lfw_total_acc)
            
    def evaluate(self,trip_loader,loss_loader,total=5000,loss_rounds=10,verbose=False):
        self.eval()
        num = 0
        if self.use_cuda:
            self.cuda()
        running_ce_loss = 0.
        iter_rounds = range(loss_rounds)
        if verbose:
            iter_rounds = tqdm(iter_rounds)
        for count in iter_rounds:
                data,label = next(iter(loss_loader))
                data,label = Variable(data,volatile=True),Variable(label,volatile=True)
                if self.use_cuda:
                    data,label = data.cuda(),label.cuda()

                prediction = self.forward(data,label)

                cross_entropy_loss = self.criterion(prediction,label)

                running_ce_loss += cross_entropy_loss.data[0]
        running_ce_loss /= loss_rounds
        
        pos_dis_list = []
        neg_dis_list = []
        while num <= total:
            try:
                pos1,pos2,neg = next(iter(trip_loader))
            except:
                continue
            num += pos1.shape[0]
            pos1 = Variable(pos1,volatile=True)
            pos2 = Variable(pos2,volatile=True)
            neg = Variable(neg,volatile=True)
            if self.use_cuda:
                pos1,pos2,neg = pos1.cuda(),pos2.cuda(),neg.cuda()
            pos1_fea = self.forward(pos1)
            pos2_fea = self.forward(pos2)
            neg_fea = self.forward(neg)
            
            pos_dis = torch.sum(pos1_fea*pos2_fea,1).data.cpu().numpy()
            neg1_dis = torch.sum(pos1_fea*neg_fea,1).data.cpu().numpy()
            neg2_dis = torch.sum(pos2_fea*neg_fea,1).data.cpu().numpy()
            
            neg_dis = np.concatenate((neg1_dis,neg2_dis),axis=0)
            pos_dis_list.append(pos_dis.reshape(-1,1))
            neg_dis_list.append(neg_dis.reshape(-1,1))
        pos_dis_all = np.concatenate(pos_dis_list)
        neg_dis_all = np.concatenate(neg_dis_list)
        pos_dis = np.mean(pos_dis_all)
        neg_dis = np.mean(neg_dis_all)
        pos_std = np.std(pos_dis_all)
        neg_std = np.std(neg_dis_all)
        threshold = pos_dis - (pos_dis-neg_dis)*pos_std/(pos_std+neg_std)
        pos_corrects = np.sum(pos_dis_all >= threshold)
        neg_corrects = np.sum(neg_dis_all < threshold)
        pos_acc = pos_corrects / np.float32(num)
        neg_acc = neg_corrects / (2. * np.float32(num))
        total_acc = (pos_corrects + neg_corrects/2)/(np.float32(num)*2)
        
        self.train()
        return total_acc,pos_acc,neg_acc,pos_dis,neg_dis,\
            pos_std,neg_std,threshold,running_ce_loss
    
    def lfw_evaluate(self,lfw_loader,total=5000,threshold=None,verbose=False):
        self.eval()
        num = 0
        if self.use_cuda:
            self.cuda()
        pos_dis_list = []
        neg_dis_list = []
        rounds = total//lfw_loader.batch_size + 1
        num = 0
        iter_rounds = range(rounds)
        if verbose:
            iter_rounds = tqdm(iter_rounds)
        for i in iter_rounds:
            try:
                pos1,pos2,neg,_,_ = next(iter(lfw_loader))
            except:
                continue
            num += pos1.shape[0]
            pos1 = Variable(pos1,volatile=True)
            pos2 = Variable(pos2,volatile=True)
            neg = Variable(neg,volatile=True)
            if self.use_cuda:
                pos1,pos2,neg = pos1.cuda(),pos2.cuda(),neg.cuda()
            pos1_fea = self.forward(pos1)
            pos2_fea = self.forward(pos2)
            neg_fea = self.forward(neg)
            
            pos_dis = torch.sum(pos1_fea*pos2_fea,1).data.cpu().numpy()
            neg1_dis = torch.sum(pos1_fea*neg_fea,1).data.cpu().numpy()
            neg2_dis = torch.sum(pos2_fea*neg_fea,1).data.cpu().numpy()
            
            neg_dis = np.concatenate((neg1_dis,neg2_dis),axis=0)
            pos_dis_list.append(pos_dis.reshape(-1,1))
            neg_dis_list.append(neg_dis.reshape(-1,1))
        pos_dis_all = np.concatenate(pos_dis_list)
        neg_dis_all = np.concatenate(neg_dis_list)
        pos_dis = np.mean(pos_dis_all)
        neg_dis = np.mean(neg_dis_all)
        pos_std = np.std(pos_dis_all)
        neg_std = np.std(neg_dis_all)
        threshold_predicted = pos_dis - (pos_dis-neg_dis)*pos_std/(pos_std+neg_std)
        if not threshold:
            threshold = threshold_predicted
        pos_corrects = np.sum(pos_dis_all >= threshold)
        neg_corrects = np.sum(neg_dis_all < threshold)
        pos_acc = pos_corrects / np.float32(num)
        neg_acc = neg_corrects / (2. * np.float32(num))
        total_acc = (pos_corrects + neg_corrects/2)/(np.float32(num)*2)

        self.train()
        return total_acc,pos_acc,neg_acc,pos_dis,neg_dis,\
            pos_std,neg_std,threshold_predicted
    
    def save_state(self,path,optimizer=None,extra=''):
        torch.save(self.state_dict(),path/('model_step{}_{}_{}.pth'.format(str(self.step),
                                                                           str(datetime.now())[:-10],
                                                                           extra)))
        if optimizer:
            torch.save(optimizer.state_dict(),path/('optimizer_step{}_{}_{}.pth'.format(str(self.step),
                                                                                     str(datetime.now())[:-10],
                                                                                     extra)))
    def load_state(self,model_path,optimizer=None,optimizer_path=None):
        self.load_state_dict(torch.load(model_path))
        if optimizer:
            optimizer.load_state_dict(torch.load(optimizer_path))    