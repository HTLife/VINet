
import torch as th
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import os
from utils import tools


import FlowNetC
from PIL import Image
import numpy as np

import flowlib

from PIL import Image


# class Vinet(nn.Module):

#     # you can also accept arguments in your model constructor
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Vinet, self).__init__()
#         self.hidden_size = hidden_size
        
#         self.lstm = nn.LSTMCell(input_size, hidden_size)
        

#     def forward(self, data, last_hidden):
#         input = th.cat((data, last_hidden), 1)
#         hidden = self.i2h(input)
#         output = self.h2o(hidden)
#         return hidden, output

class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        #self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size=24576, 
            hidden_size=64, 
            num_layers=1,
            batch_first=True)
        self.rnn.cuda()
        
        self.linear = nn.Linear(64,6)
        self.linear.cuda()
        
        checkpoint_path = '/notebooks/data/model/FlowNet2-C_checkpoint.pth.tar'
        if os.path.isfile(checkpoint_path):
            #block.log("Loading checkpoint '{}'".format(checkpoint_path))
            #checkpoint = th.load(checkpoint_path)

            checkpoint = th.load(checkpoint_path,\
                                map_location=lambda storage, loc: storage.cuda(0))

            #torch.load('tensors.pt', map_location=lambda storage, loc: storage)

            best_err = checkpoint['best_EPE']
            #model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
            #block.log("Loaded checkpoint '{}' (at epoch {})".format(checkpoint_path, checkpoint['epoch']))
        
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
        
        #return r_in
    
    
def model_out_to_flow_png(output):
    out_np = output[0].data.cpu().numpy()
    print(out_np.shape)

    #https://github.com/DediGadot/PatchBatch/blob/master/flowlib.py
    out_np = np.squeeze(out_np)
    print(out_np.shape)
    #out_np = np.swapaxes(out_np,0,2)
    out_np = np.moveaxis(out_np,0, -1)
    print(out_np.shape)

    im_arr = flowlib.flow_to_image(out_np)
    im = Image.fromarray(im_arr)
    im.save('test.png')

def main():
    with tools.TimerBlock("Initializing Datasets") as block:
        ## Load weights
        model = Combine()
        #model.training = False
        #print(model)
        #x[:,0:3,:,:]
        #input_size = (6,384,512)  # batch, timestep, ch, width, height
        
        #input_size = ( 2, 3, 384, 512)
        #summary_str = tools.summary(input_size, model)

    
        x_data_np_1 = np.array(Image.open("/notebooks/data/frame_0001.png"))
        x_data_np_2 = np.array(Image.open("/notebooks/data/frame_0002.png"))
        
        X = np.array([x_data_np_1, x_data_np_2])
        X = np.rollaxis(X,3,1)
        X = np.expand_dims(X, axis=0)
        #print(X.shape) #(1, 2, 3, 384, 512)
        X = Variable(th.from_numpy(X).type(th.FloatTensor).cuda())       
        output = model(X)
        print(output)
        #print(len(output))

        

    
        

if __name__ == '__main__':
    main()