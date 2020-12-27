from architectures.decoder import Generator as Decoder
from architectures.autoencoder import Generator as Autoencoder
#from losses.MaskedMSELoss import MaskedMSELoss
import losses.SSIMLoss as SSIMLoss
from dataset.TrackerDS_advanced import TrackerDS_advanced
import decoder_engine as decoder_engine
import engine_new as autoencoder_engine
from transforms import itk_transforms as itktransforms
from transforms import tensor_transforms as tensortransforms
from torchvision import transforms as torchtransforms
import numpy as np
import torch
import os

#TRANSFORMS
resample = itktransforms.Resample(new_spacing=[.5, .5, 1.])
tonumpy = itktransforms.ToNumpy(outputtype=np.float)
totensor = torchtransforms.ToTensor()
crop     = tensortransforms.CropToRatio(outputaspect=1.)
resize   = tensortransforms.Resize(size=(256,256), interp='BILINEAR')
rescale  = tensortransforms.Rescale(interval=(0,1))
transform = torchtransforms.Compose([resample,
                                     tonumpy,
                                     totensor,
                                     crop,
                                     resize,
                                     rescale])

ConstantRescale = tensortransforms.ConstantRescale(scaling = np.array([1/250,1/250,1/500,1.,1.,1.,1.]))

#DATASET AND DATALOADER
train_ds = TrackerDS_advanced("/home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator", mode = 'train', transform=transform, target_transform = ConstantRescale)
test_ds = TrackerDS_advanced("/home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator", mode = 'validate', transform=transform, target_transform = ConstantRescale)

# initiate indices to sample at each epoch
np.random.seed(42)
test_idxs = np.random.randint(len(test_ds[0]), size=(5,))
train_idxs = np.random.randint(len(train_ds), size=(5,))

config1 = {"mode": "deconv",
           "nchannels": (32, 32, 32, 32, 32, 32, 1),
           "kernel_size": (4, 4, 4, 4, 4, 1),
           "stride": (2, 2, 2, 2, 2, 1),
           "batchnorm2d": True,
           "dropout": False,
           "z_size": 7,
           "net_name": "AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1",
           "nlinear": (7, 32, 512, 1024, 2048),
           "image_size": (8, 8),
           #'end_with_sigmoid': True,
           }

cfgs = (config1,)

model_roots = [
                "/home/cm19/BEng_project/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/"\
                "AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/real_patient_data/refined_freezing_encoder/",]


results_roots = ["/home/cm19/BEng_project/results/AUTOENCODER/DeeperAuto/noDropout/MSELoss/train_and_validation_images/bigger_latent_image/"\
                "AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/real_patient_data/refined_freezing_encoder/"]

START_FROM_CHECPOINT = ((True, "/home/cm19/BEng_project/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/"
                               "AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/real_patient_data/pretraining/deconv/last.tar"),
                        )

#criterion = SSIMLoss.SSIM()
for cfg, model_root, results_root, CHECKPOINT in zip(cfgs, model_roots, results_roots, START_FROM_CHECPOINT):

    if CHECKPOINT[0]:
        assert os.path.exists(CHECKPOINT[1]), "ERROR: file not found: {}".format(CHECKPOINT[1])
        #load checkpoint
        checkpoint = torch.load(CHECKPOINT[1], map_location='cpu')
        #instanciate model
        if "DECODER" in model_root:
            net = Decoder(checkpoint['model_info']['model_configuration'])
            #FIX ME: find a way to merge checkpoint losses automatically
            losses = {}
            losses['train'] = checkpoint['train_loss_hist']
            losses['validate'] = checkpoint['val_loss_hist']
            net.load_state_dict(checkpoint['model_state_dict'], )
            for info in checkpoint['model_info']:
                net.update_info(info, checkpoint['model_info'][info])
            print("starting from checkpoint of: {}".format(net.get_info("model_name")))
            if "incomplete_dataset" not in model_root:
                print("\ntraining on full dataset!\n")
                decoder_engine.train_session(net, train_ds, test_ds, model_root, results_root, test_idxs, train_idxs,
                                             START_EPOCH=net.get_info("epoch"), MAX_EPOCH=200, lr=0.0002)
            else:
                print("\ntraining on an incomplete dataset!\n")
                decoder_engine.train_session(net, train_ds, test_ds, model_root, results_root, test_idxs, train_idxs,
                                             START_EPOCH=net.get_info("epoch"), MAX_EPOCH=200, lr=0.0002)

        elif "AUTOENCODER" in model_root:
            net = Autoencoder(checkpoint['model_info']['model_configuration'])
            net.load_state_dict(checkpoint['model_state_dict'], )
            for info in checkpoint['model_info']:
                net.update_info(info, checkpoint['model_info'][info])
            print("starting from checkpoint of: {}".format(net.get_info("model_name")))
            if "incomplete_dataset" not in model_root:
                print("\ntraining on full dataset!\n")
                autoencoder_engine.finetune_freezing_weights(net, {'mode': 'encoder',}, train_ds, test_ds, model_root, results_root, test_idxs, train_idxs,
                                                 START_EPOCH=net.get_info("epoch"), MAX_EPOCH=200, lr=0.0002, k2 = 10)
            else:
                print("\ntraining on an incomplete dataset!\n")
                autoencoder_engine.finetune_freezing_weights(net,{'mode': 'encoder',}, train_ds, test_ds, model_root, results_root, test_idxs, train_idxs,
                                                 START_EPOCH=net.get_info("epoch"), MAX_EPOCH=200, lr=0.0002, k2 = 10)

    else:
        if "DECODER" in model_root:
            net = Decoder(cfg)
            if "incomplete_dataset" not in model_root:
                print("\ntraining on full dataset!\n")
                decoder_engine.train_session(net, train_ds, test_ds, model_root, results_root, test_idxs, train_idxs, MAX_EPOCH=200, lr=0.0002)
            else:
                print("\ntraining on an incomplete dataset!\n")
                decoder_engine.train_session(net, train_ds, test_ds, model_root, results_root, test_idxs, train_idxs, MAX_EPOCH=200, lr=0.0002)
        elif "AUTOENCODER" in model_root:
            net = Autoencoder(cfg)
            if "incomplete_dataset" not in model_root:
                print("\ntraining on full dataset!\n")
                autoencoder_engine.multistep_training_session(net, train_ds, test_ds, model_root, results_root, test_idxs, train_idxs,  MAX_EPOCH=200)
            else:
                print("\ntraining on an incomplete dataset!\n")
                autoencoder_engine.multistep_training_session(net, train_ds, test_ds, model_root, results_root, test_idxs, train_idxs, MAX_EPOCH=200)