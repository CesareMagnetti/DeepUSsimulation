"""
script to generate a map of the tracker coordinates contained in the data, divided by class
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
plot_real_patient = False

classes = ("Thorax", "Abdomen", "Lower_Body", "Head", "Sweep")
colors = ("k","r","b","g","y","m")

fig1 = plt.figure()
fig2 = plt.figure()
ax_3D = fig1.add_subplot(111, projection='3d')
ax_2D = fig2.add_subplot(111)
#count corrupted points
count = 0
for idx in range(len(classes)):
    ds = TrackerDS_advanced(patient_root, mode = "train", filter_class=classes[idx])
    ds.remove_corrupted_indexes(ds.get_corrupted_indexes())
    #get positions
    x,y,z = [],[],[]

    for j in range(len(ds)):
        _, label = ds[j]
        #remove corrupted points
        if label[0] == 0 and label[1] == 0 and label[2] == 0:
            count+=1
            continue
        x.append(label[0])
        y.append(label[1])
        z.append(label[2])

        if j%100 == 0:
            print("{:.2f}%".format(j/len(ds)*100))

    ax_3D.scatter(x,y,z, c  = colors[idx], alpha = 0.5, label = classes[idx])
    ax_2D.scatter(y,z, c=colors[idx], alpha=0.5, label=classes[idx])

print("{} corrupted points".format(count))
ax_3D.set_xlabel("X (mm)")
ax_3D.set_ylabel("Y (mm)")
ax_3D.set_ylabel("Z (mm)")
ax_3D.set_title("latent space by class 3D")
plt.gcf()
plt.legend()
ax_2D.set_xlabel("Y (mm)")
ax_2D.set_ylabel("Z (mm)")
ax_2D.set_title("latent space by class 2D")
plt.gcf()
plt.legend()
plt.show()