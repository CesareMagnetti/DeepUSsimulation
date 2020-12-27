# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""
script to generate a map of losses for different decoder models saved to a checkpoint
"""
# Author:
# Cesare Magnetti <cesare.magnetti98@gmail.com> <cesare.magnetti@kcl.ac.uk>
#    King's College London, UK

import json
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from architectures.decoder import Generator as net
from mpl_toolkits.mplot3d import Axes3D



#######################################USEFUL FUNCTIONS#################################################################

def tuplify(thing):
    '''
    simple function to convert lists to tuples, because json does not differentiate between them and will automatically
    convert all tuples to lists
    :param thing: any object
    :return: tuplified object if object was a list
    '''
    if isinstance(thing,list): return tuple(map(tuplify, thing))
    else: return thing

def count_parameters(model):
    '''
    simple function to count number of parameters
    :param model: pytorch nn instance
    :return: number of trainable parameters
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




#######################################READ CONFIGURATIONS FROM TEXT FILE(S)############################################

model_root = "/Users/cesaremagnetti/Documents/BEng_project_repo/models/DECODER/DeeperDec/noDropout/MSELoss/" \
             "try_different_models/"
files = ["configurations1.txt",]# "configurations.txt"]
cfgs = []
for file in files:
    with open(model_root + file, "r") as f:
        cfgs += json.load(f)

#adjust the confusion caused by json between tuple and list objects
#print(cfgs)
for cfg in cfgs:
    for key in cfg:
        cfg[key] = tuplify(cfg[key])
#print(cfgs)


#######################################RETRIEVE INFORMATION FROM THE CHECKPOINTS########################################

names,last_val_loss,last_train_loss,linear_layers,conv_layers,num_param = [],[],[],[],[],[]
val_loss_hist, train_loss_hist = [],[]
for cfg in cfgs:
    #load checkpoint
    path = model_root+cfg['net_name']+"/"+cfg['mode'] + "/last.tar"
    checkpoint = torch.load(path, map_location = "cpu")
    #count trainable parameters
    model = net(cfg)
    num_param.append(count_parameters(model))
    #append model name
    names.append(checkpoint['model_info']['model_name'])
    #append validation loss history
    val_loss_hist.append(checkpoint['validation_loss_hist'])
    #append train loss history
    train_loss_hist.append(checkpoint['train_loss_hist'])
    #append last validation loss
    last_val_loss.append(checkpoint['validation_loss_hist'][-1])
    #append last training loss
    last_train_loss.append(checkpoint['train_loss_hist'][-1])
    #append number of linear layers
    linear_layers.append(len(checkpoint['model_info']['model_configuration']['nlinear'])-1)
    #append number of convolutional layers
    conv_layers.append(len(checkpoint['model_info']['model_configuration']['nchannels'])-1)

#convert to numpy for better handling
val_loss_hist=np.array(val_loss_hist)
train_loss_hist=np.array(train_loss_hist)
conv_layers=np.array(conv_layers)
linear_layers=np.array(linear_layers)
num_param=np.array(num_param)
last_train_loss=np.array(last_train_loss)
last_val_loss=np.array(last_val_loss)

##print stored information if desired
for idx in range(len(names)):
    print("name: {}   val_loss: {:.05}   train_loss: {:.05}   nlinear: {}   nconv: {} num parameters: {}".format(
              names[idx], last_val_loss[idx],last_train_loss[idx], linear_layers[idx],conv_layers[idx], num_param[idx]))



##################CREATE 2D COLOR CODED DIAGRAM OF LAST VALIDATION LOSS VS LINEAR LAYERS AND CONV LAYERS################

fig, axs = plt.subplots(1,3)
img = axs[0].scatter(conv_layers,linear_layers,c = last_val_loss, vmin = 0, vmax = 0.018)
fig.colorbar(img, ax = axs[0])
axs[0].grid(True)
axs[0].set_xlabel("convolutional layers")
axs[0].set_ylabel("linear layers")
axs[0].set_title("achieved validation loss\nMSE(" + str(r'$\bar{X}$') + "," + str(r'$X$')+ ")")
axs[0].set_yticks([l for l in range(min(linear_layers), max(linear_layers)+1)])
axs[0].set_xticks([c for c in range(min(conv_layers), max(conv_layers)+1)])


###CREATE 2D COLOR CODED DIAGRAM OF (% DIFFERENCE BETWEEN TRAIN AND VALIDATION LOSS) VS LINEAR LAYERS AND CONV LAYERS###

diff = [(last_val_loss[i]-last_train_loss[i])/last_val_loss[i]*100 for i in range(len(linear_layers))]

img = axs[1].scatter(conv_layers,linear_layers,c = diff)
fig.colorbar(img, ax = axs[1]).ax.yaxis.set_major_formatter(PercentFormatter(100, 0))
axs[1].grid(True)
axs[1].set_xlabel("convolutional layers")
axs[1].set_ylabel("linear layers")
axs[1].set_title("percentage difference between\ntraining and validation losses")
axs[1].set_yticks([l for l in range(min(linear_layers), max(linear_layers)+1)])
axs[1].set_xticks([c for c in range(min(conv_layers), max(conv_layers)+1)])


##################CREATE 2D COLOR CODED DIAGRAM OF LAST VALIDATION LOSS VS LINEAR LAYERS AND CONV LAYERS################

img = axs[2].scatter(conv_layers,linear_layers,c = last_train_loss, vmin = 0, vmax = 0.018)
fig.colorbar(img, ax = axs[2])
axs[2].grid(True)
axs[2].set_xlabel("convolutional layers")
axs[2].set_ylabel("linear layers")
axs[2].set_title("achieved train loss\nMSE(" + str(r'$\bar{X}$') + "," + str(r'$X$')+ ")")
axs[2].set_yticks([l for l in range(min(linear_layers), max(linear_layers)+1)])
axs[2].set_xticks([c for c in range(min(conv_layers), max(conv_layers)+1)])
plt.show()


################################PLOT VALIDATION LOSS VS NUMBER OF TRAINABLE PARAMETERS##################################
fig, axs = plt.subplots(1,2)

axs[0].plot(sorted(num_param),last_val_loss, c = "r")
axs[0].plot(sorted(num_param), last_train_loss, c = "k")
axs[0].set_xticks(sorted(num_param))
axs[1].plot(sorted(num_param),last_val_loss, c = "r")
axs[1].plot(sorted(num_param), last_train_loss, c = "k")
axs[1].set_xticks(sorted(num_param))
plt.xscale("log")
plt.show()
# log_scale = True
# if log_scale:
#     val_loss_hist=np.log(val_loss_hist)
#     train_loss_hist=np.log(train_loss_hist)
#
# scale = 5
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection='3d')
# ax.scatter([0,]*len(linear_layers),conv_layers*scale,linear_layers*scale, c = last_val_loss, vmin = 0, vmax = 0.018)
# for idx in range(len(linear_layers)):
#     if idx == 0:
#         ax.plot(range(200), [conv_layers[idx]*scale,]*200,
#                 scale*(train_loss_hist[idx,:]-min(train_loss_hist[idx,:]))/(max(train_loss_hist[idx,:])-min(train_loss_hist[idx,:]))
#                 +(linear_layers[idx]-1)*scale, c="k", label = 'train loss')
#         ax.plot(range(200), [conv_layers[idx]*scale,]*200,
#                 scale*(val_loss_hist[idx,:]-min(val_loss_hist[idx,:]))/(max(val_loss_hist[idx,:])-min(val_loss_hist[idx,:]))
#                 +(linear_layers[idx]-1)*scale,c="r", label = 'validation loss')
#     else:
#         ax.plot(range(200), [conv_layers[idx]*scale,]*200,
#                 scale*(train_loss_hist[idx,:]-min(train_loss_hist[idx,:]))/(max(train_loss_hist[idx,:])-min(train_loss_hist[idx,:]))
#                 +(linear_layers[idx]-1)*scale, c="k")
#         ax.plot(range(200), [conv_layers[idx]*scale,]*200,
#                 scale*(val_loss_hist[idx,:]-min(val_loss_hist[idx,:]))/(max(val_loss_hist[idx,:])-min(val_loss_hist[idx,:]))
#                 +(linear_layers[idx]-1)*scale,c="r")
# ax.grid(True)
# ax.set_xlabel("training epochs")
# ax.set_ylabel("convolutional layers")
# ax.set_zlabel("linear layers")
# ax.set_title("train and validation losses\nthrough the training epochs")
# ax.set_xticks([0,100,200])
# ax.set_zticks([l*scale for l in range(min(linear_layers), max(linear_layers)+1)])
# ax.set_yticks([c*scale for c in range(min(conv_layers), max(conv_layers)+1)])
#
# #ax.set_xtickslabels([0,100,200])
# ax.set_zticklabels([l for l in range(min(linear_layers), max(linear_layers)+1)])
# ax.set_yticklabels([c for c in range(min(conv_layers), max(conv_layers)+1)])
# plt.legend()
# plt.show()

