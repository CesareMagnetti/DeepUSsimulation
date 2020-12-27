'''
infer.py file to link to the plugin
contains: -initalize(checkpoint_dir <os.path/str>,model_name <str>, model_architecture <str>)
            this function loads the given checkpoint and initializes its architecture
          -simulate(tracker <numpy array, size  = 7>)
            this function simulates the US image corresponding to the input tracker vector (xyz + quaternion)

cesare magnetti 2019, King's College London
Requirments:
PyTorch: torch.__version__  = 1.1.0
SimpleITK: sitk.Version() = SimpleITK Version: 1.2.2 (ITK 4.13)
'''

import os
import torch
import numpy as np

# get device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def initialize(checkpoint_dir, model_name = "best_validation", model_architecture = None):
    '''

    :param checkpoint_dir: root directory containing the checkpoint we want to load
    :param model_name: the name of the model we desire to load (default = "best_validation)
    :param model_architecture: the architecture of the model we want to load, in None it will
           be inferred by the checkpoint directory path
    :return: net: model we wish to use to simulate images
    '''

    #CHECK THAT INPUT DIRECTORY EXISTS
    assert os.path.isdir(checkpoint_dir), "ERROR in initialize(): input parameter <checkpoint_dir> is not a directory!"

    #build and check path to load
    assert model_name in ("best_validation","best_training","last"), "ERROR in initialize(): unknown parameter <model_name>"
    path = os.path.join(checkpoint_dir, model_name + ".tar")
    assert os.path.isfile(path), "ERROR in initialize(): file '{}' not found!".format(path)

    #load checkpoint
    checkpoint = torch.load(path, map_location='cpu')
    KEYS = ('model_state_dict','model_info')
    for key in KEYS:
        assert 'model_state_dict' in checkpoint, "ERROR in initialize(): missing key in loaded checkpoint, <key: {}>".format(key)

    #import architecture
    ARCHITECTURES = ("DECODER", "AUTOENCODER", "VARIATIONAL_AUTOENCODER")
    if model_architecture is None:
        if ARCHITECTURES(0) in checkpoint_dir:
            from architectures.decoder import Generator
        elif ARCHITECTURES(1) in checkpoint_dir:
            from architectures.autoencoder import Generator
        elif ARCHITECTURES(2) in checkpoint_dir:
            from architectures.variational_autoencoder import Generator
    else:
        assert model_architecture in ARCHITECTURES, "ERROR in initialize(): unknown model architecture '{}'".format(model_architecture)
        if model_architecture == ARCHITECTURES(0):
            from architectures.decoder import Generator
        elif model_architecture == ARCHITECTURES(1):
            from architectures.autoencoder import Generator
        elif model_architecture == ARCHITECTURES(2):
            from architectures.variational_autoencoder import Generator

    #instanciate model
    net = Generator(checkpoint['model_info']['model_configuration']).to(device)
    #load state dict
    net.load_state_dict(checkpoint['model_state_dict'])
    #set net in evaluation mode
    net.eval()

    return net


def simulate(tracker, net):
    '''
    :param tracker: (7,) vector with tracker information: xyz coordinates and quaternion of transducer orientations <numpy.ndarray>
    :return: img: simulated tensor image <numpy.ndarray
    '''

    assert isinstance(tracker, np.ndarray), "ERROR in simulate(): input parameter <tracker> should be an instance of <numpy.ndarray>"
    assert tracker.shape == 7, "ERROR in simulate(): input parameter <tracker> should be a (7,) numpy array"

    #transform tracker to torch.tensor and move to GPU if available
    tracker = torch.from_numpy(tracker).unsqueeze(0).to(device)

    return net.decode(tracker).detach().cpu().numpy().squeeze()
