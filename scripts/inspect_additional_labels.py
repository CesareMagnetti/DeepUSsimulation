from dataset.TrackerDS_advanced import TrackerDS_advanced
from transforms import tensor_transforms as tensortransforms
import numpy as np
import torch
import sys
ConstantRescale = tensortransforms.ConstantRescale(scaling = np.array([1/250,1/250,1/500,1.,1.,1.,1.]))
RescaleToRangeElementWise = tensortransforms.RescaleToRangeElementWise
NormalizeElementWise = tensortransforms.NormalizeElementWise

phantom_root = "/home/cm19/BEng_project/data/data"
patient_root_extra = "/home/cm19/BEng_project/data/phantom_04Mar2020"
patient_root = "/home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator/"
ds = TrackerDS_advanced(patient_root, mode = 'train',additional_labels  = ('FocusDepth','SectorWidth'))
ds1 = TrackerDS_advanced(patient_root, mode = 'train',additional_labels  = ('FocusDepth','SectorWidth'))

_,label = ds[0]
max, min, temp = np.empty(label.shape[-1]),np.empty(label.shape[-1]),np.zeros((len(ds),label.shape[-1]))
max.fill(-sys.float_info.max)
min.fill(sys.float_info.max)
count=0
for idx, (_,label) in enumerate(ds):
    if idx%100 ==0:
        print("{}%".format(idx/len(ds)*100))
    if label[0] ==0 and label[1] == 0 and label[2] == 0 and label[3] ==1 and label[4] == 0 and label[5] == 0 and label[6]==0:
        count+=1
    for i,l in enumerate(label):
        temp[idx,i]+=l
#        if l<-500 and i==2:
            #count+=1
        if l>max[i]:
            max[i] = l
        if l<min[i]:
            min[i] = l

means, stds = [],[]

for i in range(label.shape[-1]):
    means.append(np.mean(temp[:,i]))
    stds.append(np.std(temp[:, i]))

print("means: {}\nstds: {}".format(means,stds))
#scaling = np.array([1/abs(M) if abs(M)>abs(m) else 1/abs(m) for M,m in zip(max,min)])
#in_range = tuple([(m,M) for m,M in zip(min,max)])
#out_range = (0,1)
print(count)
first = {"means": np.array(means), "stds": np.array(stds)}
second = {"in_range": ((-4,4),)*label.shape[-1], "out_range": (-1,1)}
ds.apply_target_transform((NormalizeElementWise,RescaleToRangeElementWise), first = first, second = second)
#ds.apply_target_transform(NormalizeElementWise, **first)

#for i in range(len(ds)):
  #  _,label = ds[i]
  #  _,label1 = ds1[i]
    #print(label1[-1],label[-1])
  #  for j in range(len(label)):
   #     a=1
   #     print("{}:\t{} ---> {}".format(ds.get_labels()[j], label1[j], label[j]))


#for i in range(max.shape[0]):
#   print("key: {}\t max: {}\t min: {}".format(ds.get_labels()[i], max[i], min[i]))



