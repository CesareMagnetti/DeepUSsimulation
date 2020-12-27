# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""
script to generate a map of the tracker coordinates contained in the data
"""
# Author:
# Cesare Magnetti <cesare.magnetti98@gmail.com> <cesare.magnetti@kcl.ac.uk>
#    King's College London, UK

import tensor_transforms  as tensortransforms
import itk_transforms as itktransforms
from torchvision import transforms as torchtransforms
from TrackerDS_advanced import TrackerDS_advanced
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D#TRANSFORMS
resample = itktransforms.Resample(new_spacing=[.5, .5, 1.])
tonumpy  = itktransforms.ToNumpy(outputtype='float')
totensor = torchtransforms.ToTensor()
crop     = tensortransforms.CropToRatio(outputaspect=1.)
resize   = tensortransforms.Resize(size=(256,256), interp='BILINEAR')
rescale  = tensortransforms.Rescale(interval=(0,1))
transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale])
ConstantRescale = tensortransforms.ConstantRescale(scaling = np.array([1/250,1/250,1/500,1.,1.,1.,1.]))
#DATASET AND DATALOADER
root = "/home/cm19/BEng_project/data/data"
root1 = "/home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator"
plot_real_patient = True
#root = "/home/cm19/BEng_project/data/data"
ds = (TrackerDS_advanced(root, mode = "validate", transform=transform),
      TrackerDS_advanced(root, mode = "validate", transform=transform, X_point=(-30,-0,-350), X_radius=30),
      TrackerDS_advanced(root, mode = "validate", transform=transform, reject_inside_radius=False, X_point=(-30,-0,-350), X_radius=30))

assert isinstance(ds,tuple), "ERROR: ds should be a tuple of datasets"
assert len(ds)<=3, "ERROR: max 3 datasets in ds, they should be the full data, the incomplete data and the removed data"

if plot_real_patient:
    ds1 = TrackerDS_advanced(root1, mode = "train", transform=transform)
    x_p, y_p, z_p = [], [], []
    for i, data in enumerate(ds1):
        _, tr = data
        if tr[2] > -320:
           continue
        x_p.append(tr[0])
        y_p.append(tr[1])
        z_p.append(tr[2])
        if i % 100 == 0:
            print("{:.2f}%".format(i / len(ds1) * 100))
if len(ds)>=1:
    x,y,z = [],[],[]
    for i,data in enumerate(ds[0]):
        _,tr = data
        if tr[2] > -150:
            continue
        x.append(tr[0])
        y.append(tr[1])
        z.append(tr[2])
        if i%500 == 0:
            print("{:.2f}%".format(i/len(ds[0])*100))
if len(ds)>=2:
    x1, y1, z1 = [],[],[]
    for i, data1 in enumerate(ds[1]):
        _,tr1 = data1
        if tr1[2] > -150:
            continue
        x1.append(tr1[0])
        y1.append(tr1[1])
        z1.append(tr1[2])
        if i%100 == 0:
            print("{:.2f}%".format(i/len(ds[0])*100))
if len(ds)>=3:
    x2, y2, z2 = [],[],[]
    for i, data2 in enumerate(ds[2]):
        _, tr2 = data2
        if tr2[2] > -150:
            continue
        x2.append(tr2[0])
        y2.append(tr2[1])
        z2.append(tr2[2])
        if i%100 == 0:
            print("{:.2f}%".format(i/len(ds[0])*100))

#2D plots
fig = plt.figure(1)
custom_xlim = (min(y)+0.2*min(y), max(y)+0.2*max(y))
custom_ylim = (min(z)+0.05*min(z), max(z)-0.05*max(z))
n = 3
i = 1
if plot_real_patient:
    custom_xlim_p = (min(y_p) + 0.2 * min(y_p), max(y_p) + 0.2 * max(y_p))
    custom_ylim_p = (min(z_p) + 0.05 * min(z_p), max(z_p) - 0.05 * max(z_p))
    n+=1
    ax = fig.add_subplot(1, n, 1)
    ax.scatter(y_p, z_p, c  = 'k', alpha = 0.5)
    plt.xlim(custom_xlim_p)
    plt.ylim(custom_ylim_p)
    plt.xlabel("Y (mm)")
    plt.ylabel("Z (mm)")
    ax.set_title("real patient dataset")
    i+=1
if len(ds)>=1:
    ax = fig.add_subplot(1, n, i)
    ax.scatter(y, z, c  = 'k', alpha = 0.5)
    plt.xlim(custom_xlim)
    plt.ylim(custom_ylim)
    plt.xlabel("Y (mm)")
    plt.ylabel("Z (mm)")
    ax.set_title("complete dataset")
    i+=1
if len(ds)>=2:
    ax = fig.add_subplot(1, n, i)
    ax.scatter(y1, z1, c  = 'k', alpha = 0.5)
    plt.xlim(custom_xlim)
    plt.ylim(custom_ylim)
    plt.xlabel("Y (mm)")
    plt.ylabel("Z (mm)")
    ax.set_title("incomplete dataset")
    i+=1
if len(ds)>=3:
    ax = fig.add_subplot(1, n, i)
    ax.scatter(y2, z2, c  = 'k', alpha = 0.5)
    plt.xlim(custom_xlim)
    plt.ylim(custom_ylim)
    plt.xlabel("Y (mm)")
    plt.ylabel("Z (mm)")
    ax.set_title("removed samples")
    i+=1
#plt.tight_layout()

#3D plots
fig2 = plt.figure(2)
custom_xlim = (min(x)+0.2*min(x), max(x)+0.2*max(x))
custom_ylim = (min(y)+0.2*min(y), max(y)+0.2*max(y))
custom_zlim = (min(z)+0.05*min(z), max(z)-0.05*max(z))
n = 3
i = 1
if plot_real_patient:
    custom_xlim_p = (min(x_p) + 0.2 * min(x_p), max(x_p) + 0.2 * max(x_p))
    custom_ylim_p = (min(y_p) + 0.2 * min(y_p), max(y_p) + 0.2 * max(y_p))
    custom_zlim_p = (min(z_p) + 0.05 * min(z_p), max(z_p) - 0.05 * max(z_p))
    n+=1
    ax = fig2.add_subplot(1, n, 1, projection='3d')
    ax.scatter(x_p, y_p, z_p, c  = 'k', alpha = 0.5)
    ax.set_xlim3d(custom_xlim_p)
    ax.set_ylim3d(custom_ylim_p)
    ax.set_zlim3d(custom_zlim_p)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("real patient dataset")
    i+=1
if len(ds)>=1:
    ax = fig2.add_subplot(1, n, i, projection='3d')
    ax.scatter(x, y, z, c  = 'k', alpha = 0.5)
    ax.set_xlim3d(custom_xlim)
    ax.set_ylim3d(custom_ylim)
    ax.set_zlim3d(custom_zlim)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("complete dataset")
    i+=1
if len(ds)>=2:
    ax = fig2.add_subplot(1, n, i, projection='3d')
    ax.scatter(x1, y1, z1, c  = 'k', alpha = 0.5)
    ax.set_xlim3d(custom_xlim)
    ax.set_ylim3d(custom_ylim)
    ax.set_zlim3d(custom_zlim)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("incomplete dataset")
    i+=1
if len(ds)>=3:
    ax = fig2.add_subplot(1, n, i, projection='3d')
    ax.scatter(x2, y2, z2, c  = 'k', alpha = 0.5)
    ax.set_xlim3d(custom_xlim)
    ax.set_ylim3d(custom_ylim)
    ax.set_zlim3d(custom_zlim)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("removed samples")
    i+=1
plt.tight_layout()

#show figures
plt.show()