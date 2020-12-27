import torch
import numpy as np
from matplotlib import pyplot as plt
import os

def gglob(path, regexp=None):
    """Recursive glob
    """
    import fnmatch
    import os
    matches = []
    if regexp is None:
        regexp = '*'
    for root, dirnames, filenames in os.walk(path, followlinks=True):
        for filename in fnmatch.filter(filenames, regexp):
            matches.append(os.path.join(root, filename))
    return matches

#DEFINE ROOT DIRECTORY
root = "/home/cm19/BEng_project/models/DECODER/DeeperDec/noDropout/MSELoss/change_linear/"
MODE = "last"
# Get filenames of all the available models of that mode
filenames = [os.path.realpath(y) for y in gglob(root, '*.*') if MODE in y]
train_losses,val_losses,layers = [],[],[]
for file in filenames:
    if "DEBUGGING" in file:
        continue
    checkpoint = torch.load(file, map_location='cpu')
    train_losses.append(checkpoint['train_loss_hist'])
    val_losses.append(checkpoint['validation_loss_hist'])
    layers.append(len(checkpoint['model_info']['model_configuration']['nlinear'])\
             + len(checkpoint['model_info']['model_configuration']['stride']))

train_losses, val_losses = np.array(train_losses), np.array(val_losses)
epochs = (50, 100,150,200,)
colors = ('b','r', 'g', 'k')
fig = plt.figure()
for epoch,color in zip(epochs,colors):
    plt.scatter(layers, train_losses[:,epoch-1], c = color, label = "train_loss_epoch {}".format(epoch))
    plt.scatter(layers, val_losses[:,epoch - 1], c = color, marker = '^', label="validation_loss_epoch {}".format(epoch))
plt.legend()
plt.xlabel("number of layers")
plt.ylabel("MSELoss(simulation, original)")
plt.show()