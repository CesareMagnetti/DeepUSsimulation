# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""
script to generate a map of the tracker coordinates contained in the data, divided by class
"""
# Author:
# Cesare Magnetti <cesare.magnetti98@gmail.com> <cesare.magnetti@kcl.ac.uk>
#    King's College London, UK

from architectures.autoencoder import Generator as Autoencoder
import torch
import tensor_transforms  as tensortransforms
import itk_transforms as itktransforms
from torchvision import transforms as torchtransforms
from TrackerDS_advanced import TrackerDS_advanced
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

#TRANSFORMS
resample = itktransforms.Resample(new_spacing=[.5, .5, 1.])
tonumpy  = itktransforms.ToNumpy(outputtype='float')
totensor = torchtransforms.ToTensor()
crop     = tensortransforms.CropToRatio(outputaspect=1.)
resize   = tensortransforms.Resize(size=(256,256), interp='BILINEAR')
rescale  = tensortransforms.Rescale(interval=(0,1))
transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale])
ConstantRescale = tensortransforms.ConstantRescale(scaling = np.array([1/250,1/250,1/500,1.,1.,1.,1.]))
#DATASET AND DATALOADER
phantom_root = "/home/cm19/BEng_project/data/data"
patient_root_extra = "/home/cm19/BEng_project/data/phantom_04Mar2020"
patient_root = "/home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator/"

model_root = "/home/cm19/BEng_project/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/" \
             "AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/real_patient_data/deconv/last.tar"

checkpoint = torch.load(model_root, map_location="cpu")
net = Autoencoder(checkpoint['model_info']['model_configuration'])
net.load_state_dict(checkpoint['model_state_dict'])

#move to gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
data_type = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
if use_cuda:
    net.to(device)

# filtered classed and associate colors for plotting
classes = ("Thorax", "Abdomen", "Lower_Body", "Head", "Sweep")
colors = ("k","r","b","g","y","m")

#make PCA object to reduce to 2 dimensions
pca = PCA(n_components=2)

#place holder variables for the latent space and its approximations
original_labels = []
reduced_original_labels = []
approximated_labels = []
reduced_approximated_labels = []

#loop through datasets and store labels
for c in classes:
    #instanciate dataset filtering wanted class
    ds = TrackerDS_advanced(patient_root, mode = "train", transform=transform, filter_class=c)
    #remove corrupted samples
    ds.remove_corrupted_indexes(ds.get_corrupted_indexes())
    #initiate label container
    N,N_features = len(ds),len(ds.get_labels())
    labels = np.zeros((N,N_features))
    approx_labels = np.zeros((N, N_features))

    for idx in range(N):
        img, label = ds[idx]
        # set the image to shape (BxCxHxW) as torch default (from ds the shape is CxHxW, B equals 1
        # because we are doing one sample each time)
        img = img.unsqueeze(0).to(device).type(data_type)
        approx_label = net.encode(img).detach().cpu().numpy().squeeze()
        #store labels
        for j in range(N_features):
            labels[idx,j] = label[j]
            approx_labels[idx, j] = approx_label[j]

        #print some info about were we are at
        if idx%100 == 0:
            print("{:.2f}%".format(idx/len(ds)*100))

    #use PCA to reduce dimensions to 2
    original_labels.append(labels)
    reduced_original_labels.append(pca.fit_transform(labels))
    approximated_labels.append(approx_labels)
    reduced_approximated_labels.append(pca.fit_transform(approx_labels))

#plot
fig = plt.figure()
fig.suptitle("Ground Truth Latent Space")
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
for i in range(len(classes)):
    ax.scatter(original_labels[i][:,1], original_labels[i][:,2], c=colors[i], label=classes[i], alpha = 0.5)
    ax1.scatter(reduced_original_labels[i][:,0],reduced_original_labels[i][:,1], c=colors[i], label=classes[i], alpha=0.5)

ax.set_title("latent space visualized in Y,Z componets")
ax.set_xlabel("Y (mm)")
ax.set_ylabel("Z (mm)")
ax1.set_title("latent space visualized using PCA")
ax1.set_xlabel("principal component 1")
ax1.set_ylabel("principale componemt 2")
plt.legend()
plt.tight_layout()

fig1 = plt.figure()
fig1.suptitle("Approximated Latent Space")
ax = fig1.add_subplot(121)
ax1 = fig1.add_subplot(122)
for i in range(len(classes)):
    ax.scatter(approximated_labels[i][:,1], approximated_labels[i][:,2], c=colors[i], label=classes[i], alpha = 0.5)
    ax1.scatter(reduced_approximated_labels[i][:,0],reduced_approximated_labels[i][:,1], c=colors[i], label=classes[i], alpha=0.5)

ax.set_title("latent space visualized in Y,Z componets")
ax.set_xlabel("Y (mm)")
ax.set_ylabel("Z (mm)")
ax1.set_title("latent space visualized using PCA")
ax1.set_xlabel("principal component 1")
ax1.set_ylabel("principale componemt 2")
plt.legend()
plt.tight_layout()
plt.show()

plt.show()