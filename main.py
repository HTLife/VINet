
import torch
import torch.nn as nn

import os
from utils import tools



import FlowNetC

def main():




    

    with tools.TimerBlock("Initializing Datasets") as block:
        ## Load weights
        checkpoint_path = '/notebooks/data/model/FlowNet2-C_checkpoint.pth.tar'
        if os.path.isfile(checkpoint_path):
            block.log("Loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)

            #torch.load('tensors.pt', map_location=lambda storage, loc: storage)

            best_err = checkpoint['best_EPE']
            #model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(checkpoint_path, checkpoint['epoch']))
        
        
        model = FlowNetC.FlowNetC(batchNorm=False)
        model.load_state_dict(checkpoint['state_dict'])
        print(model)
    

if __name__ == '__main__':
    main()