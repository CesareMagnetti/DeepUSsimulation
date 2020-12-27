from architectures.autoencoder import Generator as Autoencoder
from architectures.decoder import Generator as Decoder
from dataset.TrackerDS_advanced import TrackerDS_advanced
from engine import *
from transforms import itk_transforms as itktransforms
from transforms import tensor_transforms as tensortransforms
from torchvision import transforms as torchtransforms
import torch
import numpy as np
import random
import glob
from matplotlib import pyplot as plt
#import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
import matplotlib.backends.backend_pdf#TRANSFORMS
resample = itktransforms.Resample(new_spacing=[.5, .5, 1.])
tonumpy  = itktransforms.ToNumpy(outputtype='float')
totensor = torchtransforms.ToTensor()
crop     = tensortransforms.CropToRatio(outputaspect=1.)
resize   = tensortransforms.Resize(size=(256,256), interp='BILINEAR')
rescale  = tensortransforms.Rescale(interval=(0,1))
transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale])
ConstantRescale = tensortransforms.ConstantRescale(scaling = np.array([1/250,1/250,1/500,1.,1.,1.,1.]))#DATASET AND DATALOADER
root = "/home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator"
ds = TrackerDS_advanced(root, mode = "validate", transform=transform, target_transform=ConstantRescale)# ,reject_inside_radius=False, X_point=(-30,-0,-350), X_radius=30)#CUDA SETUP
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")#DEFINE ROOT DIRECTORY OF EACH MODEL

roots = [
                "/home/cm19/BEng_project/models/DECODER/DeeperDec/noDropout/MSELoss/bigger_latent_image/DECODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/real_patient_data/deconv/last.tar",

                "/home/cm19/BEng_project/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/real_patient_data/deconv/last.tar",

                "/home/cm19/BEng_project/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/real_patient_data/refined_freezing_encoder/deconv/last.tar",
]

GT_files_names = [1,2,3]
names = ["decoder", "autoencoder", "pretrained autoencoder"]
names1 = ["1", "2", "3"]

assert len(names) == len(roots), "ERROR: names and roots must have same length!"
nets = []
for idx, root in enumerate(roots):
    checkpoint = torch.load(root, map_location='cpu')
    cfg = checkpoint['model_info']['model_configuration']
    print(checkpoint['model_info']['epoch'])
    # cfg['interpolation_scheme'] = 'nearest'
    for key in cfg.keys():
        print("{} --> {}".format(key, cfg[key]))
    if "DECODER" in cfg['net_name']:
        net = Decoder(cfg).to(device)
    elif "AUTOENCODER" in cfg['net_name']:
        net = Autoencoder(cfg).to(device)
    else:
        net = Decoder(cfg).to(device)
        print("WARNING: inconsistent name for root directory:\n{}\n, decoder architecture was instanciated by default,"\
              " rename the root directory accordingly if needed:\n i.e. ~/ANY PATH/ARCHITECTURE TYPE (egs: DECODER or"\
              " AUTOENCODER)/ANY PATH/model.tar")    # 1. filter out unnecessary keys
    pretrained_dict = {}
    flag = True
    for k, v in checkpoint['model_state_dict'].items():
        if k not in net.state_dict():
            if flag:
                print("the following parameters will be discarded:\n")
                flag = False
            print("{}: {}".format(k, v))
        else:
            pretrained_dict[k] = v    #load state dict
    net.load_state_dict(pretrained_dict)    #net.print_info(True)
    net.eval()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    #names.append(net.get_info('model_name'))
    nets.append(net)

OUTS = []
NAMES = []
pdf = matplotlib.backends.backend_pdf.PdfPages("/home/cm19/BEng_project/patient_image_quality_GT.pdf")
pdf1 = matplotlib.backends.backend_pdf.PdfPages("/home/cm19/BEng_project/patient_image_quality_survey.pdf")#determine indices of images to show at each epoch
file = open("/home/cm19/BEng_project/patient_GT.txt", "w")
for i in range(100):
    #get random sample
    image, tracker = ds[np.random.randint(len(ds))]
    image,tracker = image.unsqueeze(0).to(device).type(torch.cuda.FloatTensor), torch.tensor(tracker).unsqueeze(0).to(device).type(torch.cuda.FloatTensor)
    image = image.detach().cpu().numpy().squeeze()
    #shuffle nets and names
    temp = list(zip(names,nets,GT_files_names))
    random.shuffle(temp)
    names,nets,GT_files_names = zip(*temp)
    outs = []
    for net in nets:
        out = net.decode(tracker)
        out = out.detach().cpu().numpy().squeeze()
        outs.append(out)

    fig, axes = plt.subplots(1, len(nets) + 1)
    axes[0].imshow(image, interpolation='nearest', cmap='gray')
    axes[0].axis('off')
    axes[0].title.set_text('original image')
    for ax, name, out in zip(axes[1:], names, outs):
        ax.imshow(out, interpolation='nearest', cmap='gray')
        ax.axis('off')
        ax.title.set_text(name)
    plt.tight_layout()
    pdf.savefig(fig)

    fig1, axes1 = plt.subplots(1, len(nets) + 1)
    plt.suptitle("{}. Which image between 1,2 and 3 resembles the original image the most?".format(i+1))
    axes1[0].imshow(image, interpolation='nearest', cmap='gray')
    axes1[0].axis('off')
    axes1[0].title.set_text('original image')
    for ax, name1, out in zip(axes1[1:], names1, outs):
        ax.imshow(out, interpolation='nearest', cmap='gray')
        ax.axis('off')
        ax.title.set_text(name1)
    plt.tight_layout()
    plt.savefig("/home/cm19/BEng_project/survey_images/{}.png".format(i+1))
    pdf1.savefig(fig1)

    file.write("{} {} {} {}\n".format(i+1,GT_files_names[0], GT_files_names[1], GT_files_names[2]))
    # OUTS.append(np.hstack(np.array([image,]+outs)))
    # NAMES.append(names)
pdf.close()
pdf1.close()
file.close()
# IMAGE = np.vstack(OUTS)
# print(NAMES)
# cmap = plt.cm.gray
# norm = plt.Normalize(vmin=0, vmax=1)
# fig = plt.figure()
# plt.imshow(cmap(norm(IMAGE)))
# plt.axis("off")
# plt.show()



