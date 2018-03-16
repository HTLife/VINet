
import torch as th
import torch.nn as nn
from torch.autograd import Variable

import os
from utils import tools


import FlowNetC


def main():
    with tools.TimerBlock("Initializing Datasets") as block:
        ## Load weights
        checkpoint_path = '/notebooks/data/model/FlowNet2-C_checkpoint.pth.tar'
        if os.path.isfile(checkpoint_path):
            block.log("Loading checkpoint '{}'".format(checkpoint_path))
            #checkpoint = th.load(checkpoint_path)

            checkpoint = th.load(checkpoint_path,\
                                map_location=lambda storage, loc: storage.cuda(0))

            #torch.load('tensors.pt', map_location=lambda storage, loc: storage)

            best_err = checkpoint['best_EPE']
            #model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(checkpoint_path, checkpoint['epoch']))
        
        
        model = FlowNetC.FlowNetC(batchNorm=False)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        #print(model)
        #x[:,0:3,:,:]
        input_size = (6,384,512)
        summary_str = tools.summary(input_size, model)
    
        

if __name__ == '__main__':
    main()