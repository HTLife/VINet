
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

import os
from utils import tools


import FlowNetC
from PIL import Image
import numpy as np

import flowlib

from PIL import Image

import csv

#dset = MyDataset('/notebooks/data/euroc/', V1_01_easy)
#loader = torch.utils.DataLoader(dset, num_workers=8)  
#https://discuss.pytorch.org/t/loading-huge-data-functionality/346/2

class MyDataset:
    
    def __init__(self):
        self.base_dir = '/notebooks/data/euroc/'
        self.sequence = 'V1_01_easy'
        self.base_path_img = self.base_dir + self.sequence + '/cam0/data/'
        self.data_files = os.listdir(self.base_dir + self.sequence + '/cam0/data/')
        self.data_files.sort()
        
        self.trajectory_abs = []  #abosolute camera pose
        with open(self.base_dir + self.sequence + '/vicon0/sampled_relative.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
#                 parsed = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), 
#                           float(row[4]), float(row[5]), float(row[6]), float(row[7])]
                parsed = [float(row[1]), float(row[2]), float(row[3]), 
                          float(row[4]), float(row[5]), float(row[6]), float(row[7])]
                self.trajectory_abs.append(parsed)
        self.trajectory_abs = np.array(self.trajectory_abs)
                
    def __len__(self):
        return len(self.data_files)
    
    def load_img(self, idx):
        x_data_np_1 = np.array(Image.open(self.base_path_img + self.data_files[idx]))
        x_data_np_2 = np.array(Image.open(self.base_path_img + self.data_files[idx+1]))
        
        
        x_data_np_1 = np.array([x_data_np_1, x_data_np_1, x_data_np_1])
        x_data_np_2 = np.array([x_data_np_2, x_data_np_2, x_data_np_2])
        
        X = np.array([x_data_np_1, x_data_np_2])

        #X = np.rollaxis(X,3,1)
        X = np.expand_dims(X, axis=0)   #(1, 2, 3, 384, 512)

        X = Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda())    
        Y = Variable(torch.from_numpy(self.trajectory_abs[idx+1]).type(torch.FloatTensor).cuda())
        return X, Y

    
    
class Vinet(nn.Module):
    def __init__(self):
        super(Vinet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=24576, 
            hidden_size=64, 
            num_layers=1,
            batch_first=True)
        self.rnn.cuda()
        
        self.linear = nn.Linear(64,7)
        self.linear.cuda()
        checkpoint = None
        checkpoint_pytorch = '/notebooks/data/model/FlowNet2-C_checkpoint.pth.tar'
        if os.path.isfile(checkpoint_pytorch):
            checkpoint = torch.load(checkpoint_pytorch,\
                                map_location=lambda storage, loc: storage.cuda(0))
            best_err = checkpoint['best_EPE']
        else:
            print('No checkpoint')

        
        self.flownetc = FlowNetC.FlowNetC(batchNorm=False)
        self.flownetc.load_state_dict(checkpoint['state_dict'])
        self.flownetc.cuda()

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size, timesteps * C, H, W)
        c_out = self.flownetc(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.linear(r_out[-1, -1, :])
        return r_out2
    
    
def model_out_to_flow_png(output):
    out_np = output[0].data.cpu().numpy()
    #print(out_np.shape)

    #https://gitorchub.com/DediGadot/PatchBatch/blob/master/flowlib.py
    out_np = np.squeeze(out_np)
    #print(out_np.shape)
    #out_np = np.swapaxes(out_np,0,2)
    out_np = np.moveaxis(out_np,0, -1)
    #print(out_np.shape)

    im_arr = flowlib.flow_to_image(out_np)
    im = Image.fromarray(im_arr)
    im.save('test.png')


def train(epoch, model, optimizer):
    model.train()
    mydataset = MyDataset()
    criterion  = nn.MSELoss()
    for i in range(len(mydataset)-1):
        data, target = mydataset.load_img(i)
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i, i , len(mydataset),
                100. * i / len(mydataset), loss.data[0]))    
    
def main():
    EPOCH = 10
    
    model = Vinet()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    #dset = MyDataset('/notebooks/data/euroc/', 'V1_01_easy')
    #train_loader = torch.utils.data.DataLoader(dset, num_workers=8)  
    train(EPOCH, model, optimizer)
          
    ## Load weights
#     model = Vinet()
#     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
#     train(EPOCH, model, optimizer)
    

#     x_data_np_1 = np.array(Image.open("/notebooks/data/frame_0001.png"))
#     x_data_np_2 = np.array(Image.open("/notebooks/data/frame_0002.png"))

#     X = np.array([x_data_np_1, x_data_np_2])
#     X = np.rollaxis(X,3,1)
#     X = np.expand_dims(X, axis=0)   #(1, 2, 3, 384, 512)
#     X = Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda())       
#     output = model(X)

        

    
        

if __name__ == '__main__':
    main()
    
        #model.training = False
    #print(model)
    #x[:,0:3,:,:]
    #input_size = (6,384,512)  # batch, timestep, ch, widtorch, height
    #input_size = ( 2, 3, 384, 512)
    #summary_str = tools.summary(input_size, model)