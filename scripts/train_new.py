from architectures.decoder import Generator as Decoder
from architectures.autoencoder import Generator as Autoencoder
from architectures.variational_autoencoder import  Generator as Variational_Autoencoder
#from losses.MaskedMSELoss import MaskedMSELoss
import losses.SSIMLoss as SSIMLoss
from dataset.TrackerDS_advanced import TrackerDS_advanced
import decoder_engine as decoder_engine
import engine_new as autoencoder_engine
import engine_variational as variational_autoencoder_engine
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

#config2 = {"mode": "deconv",
#           "nchannels": (32, 32, 32, 32, 32, 32, 1),
#           "kernel_size": (4, 4, 4, 4, 4, 1),
#           "stride": (2, 2, 2, 2, 2, 1),
#           "batchnorm2d": True,
#           "dropout": False,
#           "z_size": 7,
#           "net_name": "AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1",
#           "nlinear": (7, 32, 512, 1024, 2048),
#           "image_size": (8, 8),
#           #'end_with_sigmoid': True,
#           }

#config3 = {"mode": "deconv",
#           "nchannels": (32, 32, 32, 32, 32, 32, 1),
#           "kernel_size": (4, 4, 4, 4, 4, 1),
#           "stride": (2, 2, 2, 2, 2, 1),
#           "batchnorm2d": True,
#           "dropout": False,
#           "z_size": 7,
#           "net_name": "VARIATIONAL_AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1",
#           "nlinear": (7, 32, 512, 1024, 2048),
#           "image_size": (8, 8),
#           'end_with_sigmoid': True,
#           "BETA": 0.00001,
#           }

config1 = {"mode": "resize_conv",
           "sizes": ((16, 16), (32, 32), (64, 64), (128, 128), (256, 256)),
           "nchannels": (32, 32, 32, 32, 32, 1),
           "kernel_size": (3,3,3,3,3),
           "stride": (1,1,1,1,1),
           "batchnorm2d": True,
           "dropout": False,
           "z_size": 7,
           "net_name": "DECODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1",
           "nlinear": (7, 32, 256, 512, 1024, 2048),
           "image_size": (8, 8),
           #'end_with_sigmoid': True,
           }


cfgs = (config1,)

model_roots = [
                "/home/cm19/BEng_project/models/DECODER/DeeperDec/noDropout/MSELoss/try_different_models/"
                "extra_relu/DECODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/",
               ]

results_roots = [
                "/home/cm19/BEng_project/results/DECODER/DeeperDec/noDropout/MSELoss/train_and_validation_images/try_different_models/"
                "extra_relu/DECODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/",
                ]

phantom_root = "/home/cm19/BEng_project/data/data"
phantom_root_extra = "/home/cm19/BEng_project/data/phantom_04Mar2020"
patient_root = "/home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator/"
train_ds = TrackerDS_advanced("/home/cm19/BEng_project/data/data", mode = 'train', transform=transform, target_transform = ConstantRescale, X_point=(-30,-0,-350), X_radius=30)
test_ds = (TrackerDS_advanced("/home/cm19/BEng_project/data/data", mode = 'validate', transform=transform, target_transform = ConstantRescale,
                               X_point=(-30,-0,-350), X_radius=30),
           TrackerDS_advanced("/home/cm19/BEng_project/data/data", mode = 'validate', transform=transform, target_transform = ConstantRescale,
                              reject_inside_radius=False, X_point=(-30,-0,-350), X_radius=30))
train_ds1 = TrackerDS_advanced(phantom_root_extra, mode = 'train', transform=transform, target_transform = ConstantRescale)
            #TrackerDS_advanced("/home/cm19/BEng_project/data/phantom_04Mar2020", mode = 'train', transform=transform, target_transform = ConstantRescale),)
test_ds1 = TrackerDS_advanced(phantom_root_extra, mode = 'validate', transform=transform, target_transform = ConstantRescale)
print("datasets before filtering corrupted files:\n train set: {} samples\n test set: {} samples\n".format(len(train_ds1),len(test_ds1)))
train_ds1.remove_corrupted_indexes(train_ds1.get_corrupted_indexes())
test_ds1.remove_corrupted_indexes(test_ds1.get_corrupted_indexes())
print("datasets after filtering corrupted files:\n train set: {} samples\n test set: {} samples\n".format(len(train_ds1),len(test_ds1)))
# initiate indices to sample at each epoch
np.random.seed(42)
test_idxs = np.random.randint(len(test_ds1), size=(5,))
train_idxs = np.random.randint(len(train_ds1), size=(5,))
##flag to start from checkpoint, tuple containing flags, if true: (True, MODEL) if false: (False,)
#START_FROM_CHECPOINT = ((True, "/home/cm19/BEng_project/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/"\
#                               "AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/real_patient_data/pretraining/deconv/last.tar"),
#                        (False,), (False,), (False,))
START_FROM_CHECPOINT = ((False,),)

for cfg, model_root, results_root, CHECKPOINT in zip(cfgs, model_roots, results_roots, START_FROM_CHECPOINT):

    if CHECKPOINT[0]:
        assert os.path.exists(CHECKPOINT[1]), "ERROR: file not found: {}".format(model_root+CHECKPOINT[1])
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
                decoder_engine.train_session(net, train_ds1, test_ds1, model_root, results_root, test_idxs, train_idxs,
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
                autoencoder_engine.finetune_freezing_weights(net,{"mode": "encoder"}, train_ds1, test_ds1, model_root, results_root, test_idxs, train_idxs,
                                                 START_EPOCH=net.get_info("epoch"), MAX_EPOCH=200, lr=0.0002)
            else:
                print("\ntraining on an incomplete dataset!\n")
                autoencoder_engine.train_session(net, train_ds, test_ds, model_root, results_root, test_idxs, train_idxs,
                                                 START_EPOCH=net.get_info("epoch"), MAX_EPOCH=200, lr=0.0002)

    else:
        if "DECODER" in model_root:
            net = Decoder(cfg)
            if "incomplete_dataset" not in model_root:
                print("\ntraining on full dataset!\n")
                decoder_engine.train_session(net, train_ds1, test_ds1, model_root, results_root, test_idxs, train_idxs, MAX_EPOCH=200, lr=0.0002)
            else:
                print("\ntraining on an incomplete dataset!\n")
                decoder_engine.train_session(net, train_ds, test_ds, model_root, results_root, test_idxs, train_idxs, MAX_EPOCH=200, lr=0.0002)
        elif "AUTOENCODER" in model_root and "VARIATIONAL_AUTOENCODER" not in model_root:
            net = Autoencoder(cfg)
            if "incomplete_dataset" not in model_root:
                print("\ntraining on full dataset!\n")
                autoencoder_engine.train_session(net, train_ds1, test_ds1, model_root, results_root, test_idxs, train_idxs,  MAX_EPOCH=200)
            else:
                print("\ntraining on an incomplete dataset!\n")
                autoencoder_engine.train_session(net, train_ds, test_ds, model_root, results_root, test_idxs, train_idxs, MAX_EPOCH=200)
        elif "VARIATIONAL_AUTOENCODER" in model_root:
            net = Variational_Autoencoder(cfg)
            if "incomplete_dataset" not in model_root:
                print("\ntraining on full dataset!\n")
                variational_autoencoder_engine.train_session(net, train_ds1, test_ds1, model_root, results_root, test_idxs, train_idxs, criterion1=criterion,  MAX_EPOCH=200)
            else:
                print("\ntraining on an incomplete dataset!\n")
                variational_autoencoder_engine.train_session(net, train_ds, test_ds, model_root, results_root, test_idxs, train_idxs, MAX_EPOCH=200)


