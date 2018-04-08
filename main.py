# python2.7
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
import time



class MyDataset:
    
    def __init__(self, base_dir, sequence):
        #self.base_dir = '/notebooks/data/euroc/'
        #self.sequence = 'V1_01_easy'
        self.base_dir = base_dir
        self.sequence = sequence
        self.base_path_img = self.base_dir + self.sequence + '/cam0/data/'
        self.data_files = os.listdir(self.base_dir + self.sequence + '/cam0/data/')
        self.data_files.sort()
        
        self.trajectory_relative = []  #abosolute camera pose
        with open(self.base_dir + self.sequence + '/vicon0/sampled_relative.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                parsed = [float(row[1]), float(row[2]), float(row[3]), 
                          float(row[4]), float(row[5]), float(row[6]), float(row[7])]
                self.trajectory_relative.append(parsed)
                
        self.trajectory_relative = np.array(self.trajectory_relative)
        
        
        
        self.trajectory_abs = []  #abosolute camera pose
        with open(self.base_dir + self.sequence + '/vicon0/sampled.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                parsed = [float(row[1]), float(row[2]), float(row[3]), 
                          float(row[4]), float(row[5]), float(row[6]), float(row[7])]
                self.trajectory_abs.append(parsed)
                
        self.trajectory_abs = np.array(self.trajectory_abs)
    
    def getTrajectoryAbs(self):
        return self.trajectory_relative
    
    def __len__(self):
        return len(self.trajectory_relative)
    
    def load_img(self, idx):
        x_data_np_1 = np.array(Image.open(self.base_path_img + self.data_files[idx]))
        x_data_np_2 = np.array(Image.open(self.base_path_img + self.data_files[idx+1]))
        
        
        x_data_np_1 = np.array([x_data_np_1, x_data_np_1, x_data_np_1])
        x_data_np_2 = np.array([x_data_np_2, x_data_np_2, x_data_np_2])
        
        X = np.array([x_data_np_1, x_data_np_2])

        X = np.expand_dims(X, axis=0)   #(1, 2, 3, 384, 512)

        X = Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda())    
        Y = Variable(torch.from_numpy(self.trajectory_relative[idx+1]).type(torch.FloatTensor).cuda())
        
        return X, Y
    
    def load_img_bat(self, idx, batch):
        batch_x = []
        for i in range(batch):
            x_data_np_1 = np.array(Image.open(self.base_path_img + self.data_files[idx]))
            x_data_np_2 = np.array(Image.open(self.base_path_img + self.data_files[idx+1]))

            ## 3 channels
            x_data_np_1 = np.array([x_data_np_1, x_data_np_1, x_data_np_1])
            x_data_np_2 = np.array([x_data_np_2, x_data_np_2, x_data_np_2])

            X = np.array([x_data_np_1, x_data_np_2])
            batch_x.append(X)

            #X = np.expand_dims(X, axis=0)   #(1, 2, 3, 384, 512)  batch, time, ch, h, w
        
        batch_x = np.array(batch_x)
        
        X = Variable(torch.from_numpy(batch_x).type(torch.FloatTensor).cuda())    
        #print(self.trajectory_relative[idx+1:idx+1+batch, ...].shape)
        Y = Variable(torch.from_numpy(self.trajectory_relative[idx+1:idx+1+batch]).type(torch.FloatTensor).cuda())
        Y2 = Variable(torch.from_numpy(self.trajectory_abs[idx+1:idx+1+batch]).type(torch.FloatTensor).cuda())
        
        return X, Y, Y2

    
    
class Vinet(nn.Module):
    def __init__(self):
        super(Vinet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=24576, 
            hidden_size=64,#64, 
            num_layers=3,
            batch_first=True)
        self.rnn.cuda()
        
        self.linear = nn.Linear(64, 7)
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
        #print(x.size())
        
        c_in = x.view(batch_size, timesteps * C, H, W)
        #print(c_in.shape)
        c_out = self.flownetc(c_in)
        #print(c_out.shape)
        r_in = c_out.view(batch_size, timesteps, -1)
        #print(r_in.shape)
        r_out, (h_n, h_c) = self.rnn(r_in)
        
        out = self.linear(r_out[:,-1,:])

        return out
    
    
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


def train(epoch, model, optimizer, batch):
    model.train()

    mydataset = MyDataset('/notebooks/data/euroc/', 'V1_01_easy')
    #criterion  = nn.MSELoss()
    criterion  = nn.L1Loss(size_average=False)
    
    start = 5
    end = len(mydataset)-batch
    batch_num = (end - start) / batch
    startT = time.time() 
    with tools.TimerBlock("Start training") as block:
        for k in range(epoch):
            for i in range(start, end, batch):#len(mydataset)-1):
                data, target, target2 = mydataset.load_img_bat(i, batch)
                data, target, target2 = data.cuda(), target.cuda(), target2.cuda()

                optimizer.zero_grad()
                output = model(data)
                #print('output', output.shape)
                #print('target', target.shape)
                loss = criterion(output, target) + criterion(output, target2)

                loss.backward()
                optimizer.step()

                if i % 1 == 0:
                    avgTime = block.avg()
                    remainingTime = int((batch_num*epoch -  (i/batch + batch_num*k)) * avgTime)
                    rTime_str = "{:02d}:{:02d}:{:02d}".format(int(remainingTime/60//60), int(remainingTime//60%60), int(remainingTime%60))
                    #[16.651m] Train Epoch: 574    [574/577 (99%)] Loss: 19.724487, TimeAvg: 1.7407, Remaining: 00:00:05

                    
                    block.log('Train Epoch: {}\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}, TimeAvg: {:.4f}, Remaining: {}'.format(
                        k, i/batch , batch_num,
                        100. * (i/batch + batch_num*k) / (batch_num*epoch), loss.data[0], avgTime, rTime_str))
#             if i % 500 == 0 and i != 0:
            check_str = 'checkpoint_{}.pt'.format(k)
            torch.save(model.state_dict(), check_str)
            
    
    #torch.save(model, 'vinet_v1_01.pt')
    #model.save_state_dict('vinet_v1_01.pt')
    torch.save(model.state_dict(), 'vinet_v1_01.pt')

def test():
    checkpoint_pytorch = '/notebooks/data/vinet/vinet_v1_01.pt'
    if os.path.isfile(checkpoint_pytorch):
        checkpoint = torch.load(checkpoint_pytorch,\
                            map_location=lambda storage, loc: storage.cuda(0))
        #best_err = checkpoint['best_EPE']
    else:
        print('No checkpoint')
    

    model = Vinet()
    model.load_state_dict(checkpoint)  
    model.cuda()
    model.eval()
    mydataset = MyDataset('/notebooks/data/euroc/', 'V2_01_easy')
    
    err = 0
    ans = []
    #for i in range(len(mydataset)-1):
    for i in range(100):
        data, target, target2 = mydataset.load_img_bat(i, 1)
        data, target, target2 = data.cuda(), target.cuda(), target2.cuda()

        output = model(data)
        
        err += float(((target - output) ** 2).mean())
        
        output = output.data.cpu().numpy()
        ans.append(output[0])
        print(output[0])
        
        print('{}/{}'.format(str(i+1), str(len(mydataset)-1)) )
    print('err = {}'.format(err/(len(mydataset)-1)))  
    
    trajectory_relative = mydataset.getTrajectoryAbs()
    print(trajectory_relative[0])
    x = trajectory_relative[0].astype(str)
    x = ",".join(x)
    
    with open('/notebooks/data/euroc/V2_01_easy/vicon0/sampled_relative_ans.csv', 'w+') as f:
        tmpStr = x
        f.write(tmpStr + '\n')        
        
        for i in range(len(ans)-1):
            tmpStr = ans[i].astype(str)
            tmpStr = ",".join(tmpStr)
            print(tmpStr)
            print(type(tmpStr))
            f.write(tmpStr + '\n')      
    
def main():
    EPOCH = 10
    BATCH = 5
    model = Vinet()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    train(EPOCH, model, optimizer, BATCH)
          
    #test()

    
        

if __name__ == '__main__':
    main()
    
        #model.training = False
    #print(model)
    #x[:,0:3,:,:]
    #input_size = (6,384,512)  # batch, timestep, ch, widtorch, height
    #input_size = ( 2, 3, 384, 512)
    #summary_str = tools.summary(input_size, model)
