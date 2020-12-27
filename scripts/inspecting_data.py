from dataset.TrackerDS_advanced import TrackerDS_advanced
from transforms import tensor_transforms as tensortransforms
from torchvision import transforms as torchtransforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
ConstantRescale = tensortransforms.ConstantRescale(scaling = np.array([1/250,1/250,1/500,1.,1.,1.,1.,1/7.53, 1/98]))
RescaleToRangeElementWise = tensortransforms.RescaleToRangeElementWise(in_range=((-3.5,3.5),)*9, out_range=(0,1))
RescaleToRangeElementWise1 = tensortransforms.RescaleToRangeElementWise(in_range=((-250,250),(-250,250),(-500,0),
                                                                                  ((0,1),)*4,(5,7.522866),(35.968586,97.761909)),
                                                                        out_range=(0,1))
NormalizeElementWise = tensortransforms.NormalizeElementWise(means = np.array([-84.11274691559501, -93.4254690253933,
                                                             -377.46586526535674, 0.41029077213298265, 0.40904803165821796,
                                                             0.009191669624659128, 0.38697745679726636, 6.459984990012403,
                                                             58.41509867621906]),
                                                             stds = np.array([37.41139826076664, 56.442578499693326,
                                                             131.5519840128939, 0.22638645432344062, 0.42134262953461576,
                                                             0.13087924896421618, 0.5182618404489842, 0.45971127134175255,
                                                             12.194358524109209]))

phantom_root = "/home/cm19/BEng_project/data/data"
patient_root_extra = "/home/cm19/BEng_project/data/phantom_04Mar2020"
patient_root = "/home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator/"
ds = TrackerDS_advanced(patient_root, mode = 'train', target_transform=None, additional_labels  = ('FocusDepth','SectorWidth'))
ds1 = TrackerDS_advanced(patient_root, mode = 'train', target_transform=torchtransforms.Compose([NormalizeElementWise,
                                                                                                    RescaleToRangeElementWise]),
                         additional_labels= ('FocusDepth','SectorWidth'))

assert len(ds) == len(ds1)
_,label = ds[0]
_,label1 = ds1[0]

assert len(label) == len(label1)
labels = np.zeros((len(ds), len(label)))
labels1 = np.zeros((len(ds1), len(label1)))

for idx in range(len(ds)):
    if idx%100 ==0:
        print("{}%".format(idx/len(ds)*100))
    _,label = ds[idx]
    _, label1 = ds1[idx]
    for j in range(len(label)):
        labels[idx,j] = label[j]
        labels1[idx, j] = label1[j]

fig, axs = plt.subplots(3,3)
for idx,ax in enumerate(axs.ravel()):
    ax.hist(labels[:,idx], color="k", alpha = 0.4)
    ax.set_xlabel(ds.get_labels()[idx])
plt.tight_layout()

fig1, axs = plt.subplots(3,3)
for idx,ax in enumerate(axs.ravel()):
    ax.hist(labels1[:,idx], color="k", alpha = 0.4)
    ax.set_xlabel(ds.get_labels()[idx])

plt.tight_layout()
plt.show()