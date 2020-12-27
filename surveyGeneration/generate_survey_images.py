from architectures.autoencoder import Generator as Autoencoder
from architectures.decoder import Generator as Decoder
from architectures.variational_autoencoder import Generator as Variational_Autoencoder
from dataset.TrackerDS_advanced import TrackerDS_advanced
from transforms import itk_transforms as itktransforms
from transforms import tensor_transforms as tensortransforms
from torchvision import transforms as torchtransforms
import torch
import numpy as np
import random
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

ConstantRescale = tensortransforms.ConstantRescale(scaling = np.array([1/250,1/250,1/500,1.,1.,1.,1.]))

#DATASET AND DATALOADER
root = "/Users/cesaremagnetti/Documents/BEng_project/data/data"
ds = TrackerDS_advanced(root, mode = "validate", transform=transform, target_transform=ConstantRescale)# ,reject_inside_radius=False, X_point=(-30,-0,-350), X_radius=30)#CUDA SETUP

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#DEFINE ROOT DIRECTORY OF EACH MODEL
decoder_root = "/Users/cesaremagnetti/Documents/BEng_project_repo/models/DECODER/DeeperDec/noDropout/MSELoss/bigger_latent_image/" \
               "DECODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/deconv/last.tar"
autoencoder_root = "/Users/cesaremagnetti/Documents/BEng_project_repo/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/" \
                   "AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/deconv/last.tar"
pretrained_root = "/Users/cesaremagnetti/Documents/BEng_project_repo/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/" \
                "AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/pretrained/deconv/last.tar"
variational_root = ""


def load_state_dict(net, state_dict):
    '''
    simple function to load and check state dict
    :param net: pytorch network
    :param state_dict: state dict to load
    '''
    pretrained_dict = {}
    flag = True
    for k, v in state_dict.items():
        if k not in net.state_dict():
            if flag:
                print("the following parameters will be discarded:\n")
                flag = False
            print("{}: {}".format(k, v))
        else:
            pretrained_dict[k] = v

    #load state dict
    net.load_state_dict(pretrained_dict)
    net.eval()


#load models#######

checkpoint = torch.load(decoder_root, map_location='cpu')
cfg = checkpoint['model_info']['model_configuration']
decoder = Decoder(cfg).to(device)
load_state_dict(decoder, checkpoint['model_state_dict'])

checkpoint = torch.load(autoencoder_root, map_location='cpu')
cfg = checkpoint['model_info']['model_configuration']
autoencoder = Autoencoder(cfg).to(device)
load_state_dict(autoencoder, checkpoint['model_state_dict'])

checkpoint = torch.load(pretrained_root, map_location='cpu')
cfg = checkpoint['model_info']['model_configuration']
pretrained = Autoencoder(cfg).to(device)
load_state_dict(pretrained, checkpoint['model_state_dict'])


#generate images#######

#get array containing which architecture to use at each iteration
order_array = np.loadtxt(order_file_root)
names = ["1", "2"]

for i,idx in enumerate(order_array):
    #get random sample
    image, tracker = ds[np.random.randint(len(ds))]
    tracker = torch.tensor(tracker).unsqueeze(0).to(device).type(torch.FloatTensor)
    image = image.detach().cpu().numpy().squeeze()

    # get simulated image
    if idx == 1:
        out = decoder.decode(tracker).detach().cpu().numpy().squeeze()
    elif idx ==2:
        out = autoencoder.decode(tracker).detach().cpu().numpy().squeeze()
    elif idx == 3:
        out = pretrained.decode(tracker).detach().cpu().numpy().squeeze()

    outs = [gaussian_filter(out, sigma = 2), gaussian_filter(image, sigma = 4)]
    random.shuffle(out)

    #generate image
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(image, interpolation='nearest', cmap='gray')
    axes[0].axis('off')
    axes[0].title.set_text('original image')
    for ax, name, out in zip(axes[1:], names, outs):
        ax.imshow(out, interpolation='nearest', cmap='gray')
        ax.axis('off')
        ax.title.set_text(name)
    plt.tight_layout()
    plt.suptitle("{}. Which image between 1,2 resembles the original image the most?".format(i+1))

    plt.savefig("/Users/cesaremagnetti/Documents/BEng_project/surveys/phantom/survey_for_image_quality/survey_images/{}.png".format(i+1))



