
import torch as th
import torch.nn as nn
from torch.autograd import Variable

import os
from utils import tools

from collections import OrderedDict 

import FlowNetC



def summary(input_size, model):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

                params = 0
                if hasattr(module, 'weight'):
                    params += th.prod(th.LongTensor(list(module.weight.size())))
                    if module.weight.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False
                if hasattr(module, 'bias'):
                    params +=  th.prod(th.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = params
                
            if not isinstance(module, nn.Sequential) and \
               not isinstance(module, nn.ModuleList) and \
               not (module == model):
                hooks.append(module.register_forward_hook(hook))
                
        dtype = th.cuda.FloatTensor
        
        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(th.rand(1,*in_size)).type(dtype) for in_size in input_size]
        else:
            x = Variable(th.rand(1,*input_size)).type(dtype)
            
            
        print(x.shape)
        print(type(x[0]))
        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        model.apply(register_hook)
        # make a forward pass
        model(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        print('----------------------------------------------------------------')
        line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shpae', 'Param #')
        print(line_new)
        print('================================================================')
        total_params = 0
        trainable_params = 0
        for layer in summary:
            ## input_shape, output_shape, trainable, nb_params
            line_new = '{:>20}  {:>25} {:>15}'.format(layer, summary[layer]['output_shape'], summary[layer]['nb_params'])
            total_params += summary[layer]['nb_params']
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['nb_params']
            print(line_new)
        print('================================================================')
        print('Total params: ' + str("{:,}".format(total_params)))
        print('Trainable params: ' + str("{:,}".format(trainable_params)))
        print('Non-trainable params: ' + str("{:,}".format(total_params - trainable_params)))
        print('----------------------------------------------------------------')
        return summary


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
        summary_str = summary(input_size, model)
    
        

if __name__ == '__main__':
    main()