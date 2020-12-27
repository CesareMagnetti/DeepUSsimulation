import torch
from matplotlib import pyplot as plt
import numpy as np
from torch.backends import cudnn
import os
import sys

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
tensortype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# train
def train_epoch(net, optim, loader, epoch, criterion1, criterion2, k1, k2):
    '''
    function to train a network
    args:
      -net <torch.nn.Module>: network to train
      -optims <tuple>: tuple containing corresponding optimizers for each network
      -loader <torch.utils.data.dataloader>: data loader object containing LABELLED data
      -epoch <int>: training epoch the nets currently are
      -criterion1 <torch.nn>: criterion on autoencoded image
      -criterion2 <torch.nn>: criterion on latent space
      -k1 <float>: weight of criterion 1 (default 1.)
      -k2 <float>: weight of criterion 2 (default 0.)
    '''

    losses = {'autoencoder': [], 'encoder': [], 'decoder':[]}

    print("\ntraining MODEL: {}\n\n".format(net.get_info('model_name')))
    net.update_info("epoch", epoch)
    net.train()
    running_loss = 0
    running_loss_enc = 0
    running_loss_dec = 0
    running_loss_auto = 0

    for batch_idx, (img, target) in enumerate(loader):
        img = img.to(device).type(tensortype)
        target = target.to(device).type(tensortype)

        # forward pass
        out1, z = net(img)
        out2 = net.decode(target)
        loss_auto = k1 * criterion1(out1, img)
        loss_dec = k1 * criterion1(out2, img)
        loss_enc = k2 * criterion2(z, target)
        loss = loss_auto + loss_enc
        # backward pass
        optim.zero_grad()
        loss.backward()
        plot_grad_flow(net.named_parameters())
        optim.step()
        running_loss += loss.item()
        running_loss_auto += loss_auto.item()
        running_loss_dec += loss_dec.item()
        running_loss_enc += loss_enc.item()
        # print some information about where we are in training
        if batch_idx % 10 == 0:
            print("--> {}%\trunning_loss_dec: {:.5f}\trunning_loss_enc: {:.5f}\trunning_loss_auto: {:.5f}".format(batch_idx * 100 // len(loader), running_loss_dec, running_loss_enc, running_loss_auto))
    # store average loss
    net.update_info('train_loss', running_loss/ len(loader))
    losses['autoencoder'].append(running_loss_auto/ len(loader))
    losses['decoder'].append(running_loss_dec/ len(loader))
    losses['encoder'].append(running_loss_enc/ len(loader))
    return losses


# validate
def validate_epoch(net, loader, epoch, criterion1, criterion2, k1, k2, update_loss = True):
    '''
    function to validate a tuple of networks (and if needed optoimizers) over a given testing loader
    args:
      -net <torch.nn.Module>: network to validate
      -loader <torch.dataloader>: data loader object containing LABELLED data
      -epoch <int>: training epoch the nets currently are
      -criterion1 <torch.nn>: criterion on autoencoded image
      -criterion2 <torch.nn>: criterion on latent space
      -k1 <float>: weight of criterion 1 (default 1.)
      -k2 <float>: weight of criterion 2 (default 0.)
      -update_loss <bool>: flag to execute: net.update_info('val_loss', running_loss / len(loader))
    '''

    losses = {'autoencoder': [], 'encoder': [], 'decoder': []}

    print("\nvalidating MODEL: {}\n\n".format(net.get_info('model_name')))
    net.update_info("epoch", epoch)
    net.eval()
    running_loss = 0
    running_loss_auto = 0
    running_loss_dec = 0
    running_loss_enc = 0
    for batch_idx, (img, target) in enumerate(loader):
        img = img.to(device).type(tensortype)
        target = target.to(device).type(tensortype)
        # forward pass
        out1, z = net(img)
        out2 = net.decode(target)
        loss_auto = k1 * criterion1(out1, img)
        loss_dec = k1 * criterion1(out2, img)
        loss_enc = k2 * criterion2(z, target)
        loss = loss_auto + loss_enc
        running_loss += loss.item()
        running_loss_auto += loss_auto.item()
        running_loss_dec += loss_dec.item()
        running_loss_enc += loss_enc.item()
        # print some information about where we are in training
        if batch_idx % 10 == 0:
            print("--> {}%\trunning_loss_dec: {:.5f}\trunning_loss_enc: {:.5f}\trunning_loss_auto: {:.5f}".format(batch_idx * 100 // len(loader), running_loss_dec, running_loss_enc, running_loss_auto))
    #update model loss if needed
    if update_loss:
        net.update_info('val_loss', running_loss/ len(loader))
    # store average loss
    losses['autoencoder'].append(running_loss_auto/ len(loader))
    losses['decoder'].append(running_loss_dec/ len(loader))
    losses['encoder'].append(running_loss_enc/ len(loader))
    return losses


def store_losses(train_loss, validate_loss, losses, idx, validate_loss1 = None):

    '''
    function to assemble the losses in a input dictionary

    :param train_loss: dict containg train losses evaluated by the train_epoch() function (in this file)
    :param validate_loss: dict containg validation losses evaluated by the train_epoch() function (in this file)
    :param losses: input losses dictionary (see above example)
    :param idx: index relating to the network currently being trained (if train_epoch() and validate_epoch()
                are used to train tuples of networks)
    :param validate_loss1: dict containg additional validation losses evaluated by the train_epoch() function (in this file) (default: None)

    EXAMPLE TO CONSTRUCT INPUT LOSSES DICTIONARY:

    losses = {}
    losses['train'] = {'img': {'from_img': np.zeros((1, MAX_EPOCH - START_EPOCH)),
                               'from_z': np.zeros((1, MAX_EPOCH - START_EPOCH))},
                       'z': np.zeros((1, MAX_EPOCH - START_EPOCH))}
    losses['validate'] = {'img': {'from_img': np.zeros((1, MAX_EPOCH - START_EPOCH)),
                                  'from_z': np.zeros((1, MAX_EPOCH - START_EPOCH))},
                          'z': np.zeros((1, MAX_EPOCH - START_EPOCH))}

    or

    losses['validate'] = {'A':{'img': {'from_img': np.zeros((1, MAX_EPOCH - START_EPOCH)),
                              'from_z': np.zeros((1, MAX_EPOCH - START_EPOCH))},
                           'z': np.zeros((1, MAX_EPOCH - START_EPOCH))},

                      'B': {'img': {'from_img': np.zeros((1, MAX_EPOCH - START_EPOCH)),
                                    'from_z': np.zeros((1, MAX_EPOCH - START_EPOCH))},
                           'z': np.zeros((1, MAX_EPOCH - START_EPOCH))}
                      }
    '''
    if validate_loss1 is not None:
        for n, (tl_img1, vl_img1, tl_img2, vl_img2, tl_z, vl_z, vl1_img1,vl1_img2,vl1_z) in enumerate(
                zip(train_loss['img']['from_img'], validate_loss['img']['from_img'],
                    train_loss['img']['from_z'], validate_loss['img']['from_z'],
                    train_loss['z'], validate_loss['z'], validate_loss1['img']['from_img'],validate_loss1['img']['from_z'],validate_loss1['z'])):

            losses['train']['img']['from_img'][n][idx] = tl_img1
            losses['validate']['A']['img']['from_img'][n][idx] = vl_img1
            losses['validate']['B']['img']['from_img'][n][idx] = vl1_img1

            losses['train']['img']['from_z'][n][idx] = tl_img2
            losses['validate']['A']['img']['from_z'][n][idx] = vl_img2
            losses['validate']['B']['img']['from_z'][n][idx] = vl1_img2

            losses['train']['z'][n][idx] = tl_z
            losses['validate']['A']['z'][n][idx] = vl_z
            losses['validate']['B']['z'][n][idx] = vl1_z
    else:
        for n, (tl_img1, vl_img1, tl_img2, vl_img2, tl_z, vl_z) in enumerate(
                zip(train_loss['img']['from_img'], validate_loss['img']['from_img'],
                    train_loss['img']['from_z'], validate_loss['img']['from_z'],
                    train_loss['z'], validate_loss['z'])):
            losses['train']['img']['from_img'][n][idx] = tl_img1
            losses['validate']['img']['from_img'][n][idx] = vl_img1
            losses['train']['img']['from_z'][n][idx] = tl_img2
            losses['validate']['img']['from_z'][n][idx] = vl_img2
            losses['train']['z'][n][idx] = tl_z
            losses['validate']['z'][n][idx] = vl_z



def save(net,losses,optim):
    '''
    function to save network to a checkpoint, three copies of the network are being saved:
    last.tar, best_training.tar, best_validation.tar

    :param net: <torch.nn.Module> network
    :param optim: <torch.optim> optimizer
    :param losses: <dict> containing training  losses and validation losses


    EXAMPLE TO CONSTRUCT INPUT LOSSES DICTIONARY:

    losses = {}

    losses['train'] = {'encoder': [], 'decoder': [], 'autoencoder': []}

    if isinstance(test_ds, tuple):
        losses['validate'] = {'A': {'encoder': [], 'decoder': [], 'autoencoder': []},
                              'B': {'encoder': [], 'decoder': [], 'autoencoder': []}
                              }
    '''

    # save last epoch
    print("SAVING AS LAST TRAINING MODEL: {}".format(net.get_info('model_name')))
    savepath = net.get_info('model_root') + net.get_info('model_configuration')['mode'] + "/last.tar"
    torch.save({'model_state_dict': net.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'validation_loss_hist': losses['validate'],
                'train_loss_hist': losses['train'],
                'model_info': net.get_info(),
                }, savepath)
    # save as best validation model
    if net.get_info("val_loss") < net.get_info("best_val_loss"):
        print("SAVING AS BEST VALIDATION MODEL: {}".format(net.get_info('model_name')))
        net.update_info("best_val_loss", net.get_info("val_loss"))
        savepath = net.get_info('model_root') + net.get_info('model_configuration')['mode'] + "/best_validation.tar"
        torch.save({'model_state_dict': net.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'validation_loss_hist': losses['validate'],
                    'train_loss_hist': losses['train'],
                    'model_info': net.get_info(),
                    }, savepath)
    # save as best training model
    if net.get_info("train_loss") < net.get_info("best_train_loss"):
        print("SAVING AS BEST TRAINING MODEL: {}".format(net.get_info('model_name')))
        net.update_info("best_train_loss", net.get_info("train_loss"))
        savepath = net.get_info('model_root') + net.get_info('model_configuration')['mode'] + "/best_training.tar"
        torch.save({'model_state_dict': net.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'validation_loss_hist': losses['validate'],
                    'train_loss_hist': losses['train'],
                    'model_info': net.get_info(),
                    }, savepath)


def plot_grad_flow(named_parameters):
    '''
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    plot_grad_flow(self.model.named_parameters()) to visualize the gradient flow
    '''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")

def save_minibatch(net, ds, idxs, epoch=None, save=False):
    '''
    function to save results of multiple network on a random minibtch of the loader

    EXPECT THE FOLLOWING DIRECTORY STRUCTURE (inside the results_root directory path (model attribute)):
    /deconv or /resize_conv  -->  /train or /validate  --> images are saved here as EPOCH_XXX.png or TEST_IMAGE.png

    :param net: <torch.nn.Module> network
    :param ds: <torch.utils.dataset> dataset containing images
    :param idxs: <np.array> array of indexes to sample from ds
    :param epoch (optional): <int> current training epoch
    :param save: (optional): <bool> flag indicating if saving the image
    '''

    if epoch is not None:
        if epoch < 10:
            savepath = net.get_info('results_root') + net.get_info('model_configuration')['mode'] + "/" + \
                       ds.get_mode() + "/EPOCH_00{}.png".format(epoch)
        elif epoch >= 10 and epoch < 100:
            savepath = net.get_info('results_root') + net.get_info('model_configuration')['mode'] + "/" +\
                       ds.get_mode() + "/EPOCH_0{}.png".format(epoch)
        else:
            savepath = net.get_info('results_root') + net.get_info('model_configuration')['mode'] + "/" + \
                       ds.get_mode() + "/EPOCH_{}.png".format(epoch)
    else:
        savepath = net.get_info('results_root') + net.get_info('model_configuration')['mode'] + "/" + \
                   ds.get_mode() + "/TEST_IMAGE.png".format(epoch)
    imgs, outs, outs1 = [], [], []
    for idx in idxs:
        img, target = ds[idx]
        img = img.unsqueeze(0).to(device).type(tensortype)
        target = torch.tensor(target).unsqueeze(0).to(device).type(tensortype)

        #autoencoded image
        try:
            out,_ = net(img)
        except:
            out,_,_ = net(img)

        out = out.detach().cpu().numpy().squeeze()
        outs.append(out)
        #decoded image
        out1 = net.decode(target)
        out1 = out1.detach().cpu().numpy().squeeze()
        outs1.append(out1)
        #original image
        img = img.detach().cpu().numpy().squeeze()
        imgs.append(img)

    imgs, outs, outs1 = np.array(imgs), np.array(outs), np.array(outs1)
    IM = np.hstack(np.array(imgs))
    OUT = np.hstack(np.array(outs))
    OUT1 = np.hstack(np.array(outs1))
    IMAGE = np.vstack([np.vstack(np.array([IM, OUT])),OUT1])

    if save:
        # save image to drive
        cmap = plt.cm.gray
        norm = plt.Normalize(vmin=0, vmax=1)
        plt.imsave(savepath, cmap(norm(IMAGE)))
    outs += (out,)


#function to adjust learning rate:
def adjust_learning_rate(optimizer,lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_session(net, train_ds,test_ds, model_root, results_root, checkpoint = None, test_idxs=None, train_idxs=None, k1=1., k2=1., criterion1=None, criterion2=None,
                  START_EPOCH=0, MAX_EPOCH=500, lr=None, adjust_lr=False,batch_size=(64,16), num_workers=(8,2), shuffle=(True,False)):
    '''
    function to fully train, validate and save a network (along as some epoch results) given datasets and desired root directories

    :param net: <torch.nn.Module> network to train
    :param train_ds: <torch.utils.data.Dataset> training dataset (images + trackers)
    :param test_ds:  <torch.utils.data.Dataset or tuple of torch.utils.data.Dataset> test dataset(s) (images + trackers)
    :param model_root: <str> path to where to save the model
    :param results_root: <str> path to directory to save epochs' results
    :param checkpoint: <dict> torch checkpoint from which to resume training
    :param test_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of test_ds that we want to plot (with 'save_minibatch()')
    :param train_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of train_ds that we want to plot (with 'save_minibatch()')
    :param k1: <float> weight on criterion1 (default = 1)
    :param k2: <float> weight on criterion2 (default = 1)
    :param criterion1: <torch.nn.loss> criterion on image (default = None, default set later)
    :param criterion2: <torch.nn.loss> weight on criterion2 (default = None, default set later)
    :param START_EPOCH: <int> starting training epoch (default = 0)
    :param MAX_EPOCH: <int> max training epoch (default = 500)
    :param lr: <int> learning rate (default = None, so that optimizer uses its default value)
    :param adjust_lr: <bool> flag indicating if to use function adjust_learning_rate() (default = False)
    :param batch_size: <tuple<int>> tuple with batch sizes for train and test loaders (default = (64,16))
    :param num_workers: <tuple<int>> tuple with number of workers to load train and test datasets (default = (8, 2))
    :param shuffle: <tuple<bool>> tuple indicating whether to shuffle or not train and test loaders (default = (True, False))
    '''

    #check if model_root and results_root directories exist
    # (TO DO: check structure of results root, 'save_minibatch()' expects a particular directory sructure)
    assert os.path.isdir(model_root), "ERROR: directory to save the model <model_root: {}> not found.".format(model_root)
    assert os.path.isdir(results_root), "ERROR: directory to save results <results_root: {}> not found.".format(results_root)

    #determine indices of images to show at each epoch
    np.random.seed(42)
    if test_idxs is None:
        if isinstance(test_ds, tuple):
            test_idxs = np.random.randint(len(test_ds[0]), size=(5,))
        else:
            test_idxs = np.random.randint(len(test_ds), size=(5,))
    if train_idxs is None:
        train_idxs = np.random.randint(len(train_ds), size=(5,))
    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size[0], num_workers=num_workers[0], shuffle = shuffle[0])
    if isinstance(test_ds, tuple):
        assert len(test_ds) == 2, "ERROR in train_session: currently developed for either 1 or 2 test datasets."
        test_loader = torch.utils.data.DataLoader(test_ds[0], batch_size=batch_size[1], num_workers=num_workers[1], shuffle = shuffle[1])
        test_loader1 = torch.utils.data.DataLoader(test_ds[1], batch_size=batch_size[1], num_workers=num_workers[1], shuffle = shuffle[1])
    else:
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size[1], num_workers=num_workers[1], shuffle = shuffle[1])

    #instanciate criterions if None
    if criterion1 is None:
        criterion1 = torch.nn.MSELoss()
    if criterion2 is None:
        criterion2 = torch.nn.MSELoss()

    #instanciate optimizer
    if lr is not None:
        optim = torch.optim.Adam(net.parameters(),lr = lr)
    else:
        optim = torch.optim.Adam(net.parameters())

    # declare losses dict
    losses = {}

    if checkpoint is None:
        #setup training variables
        net.update_info('model_root', model_root)
        net.update_info("results_root", results_root)
        net.update_info("epoch", START_EPOCH)
        net.update_info("best_val_loss", sys.float_info.max)
        net.update_info("best_train_loss", sys.float_info.max)
        net.update_info("val_loss", sys.float_info.max)
        net.update_info("train_loss", sys.float_info.max)
        net.update_info("k1", k1)
        net.update_info("k2", k2)

        net.to(device)
        #print info before starting the training loop
        net.print_info(verbose=True)

        losses['train'] = {'encoder': [], 'decoder': [], 'autoencoder': []}
        if isinstance(test_ds, tuple):
            losses['validate'] = {'A': {'encoder': [], 'decoder': [], 'autoencoder': []},
                                  'B': {'encoder': [], 'decoder': [], 'autoencoder': []}
                                  }
        else:
            losses['validate'] = {'encoder': [], 'decoder': [], 'autoencoder': []}
    else:
        #restart from checkpoint
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device)
        optim.load_state_dict(checkpoint['optim_state_dict'])
        for key in checkpoint['model_info']:
            net.update_info(key,checkpoint['model_info'][key])
        losses['train'] = checkpoint['train_loss_hist']
        losses['validate'] = checkpoint['validation_loss_hist']

        #print info before starting the training loop
        net.print_info(verbose=True)

    #move to GPU
    if use_cuda:
        train_loader.pin_memory = True
        test_loader.pin_memory = True
        if isinstance(test_ds, tuple):
            test_loader1.pin_memory = True
        cudnn.benchmark = True
        criterion1.cuda
        criterion2.cuda

    # loop through epochs
    for idx, epoch in enumerate(range(net.get_info("epoch") + 1, MAX_EPOCH + 1)):
        print("\n\t\t\t**EPOCH: [{}/{}]**\n".format(epoch, MAX_EPOCH))
        #train epoch
        train_loss = train_epoch(net, optim, train_loader, epoch, criterion1, criterion2, k1=k1, k2=k2)
        #append train losses
        for key in train_loss:
            losses['train'][key].append(train_loss[key])

        #validate epoch
        validate_loss = validate_epoch(net, test_loader, epoch, criterion1, criterion2, k1=k1, k2=k2)
        #extra validate epoch if needed
        if isinstance(test_ds, tuple):
            validate_loss1 = validate_epoch(net, test_loader1, epoch, criterion1, criterion2, k1=k1, k2=k2, update_loss=False)
        #append validation losses
        if isinstance(test_ds, tuple):
            for key, key1 in zip(validate_loss, validate_loss1):
                losses['validate']['A'][key].append(validate_loss[key])
                losses['validate']['B'][key1].append(validate_loss1[key1])
        else:
            for key in validate_loss:
                losses['validate'][key].append(validate_loss[key])

        #save model
        save(net, losses, optim)
        #print epoch summary
        net.training_summary()

        if adjust_lr and lr > 10e-6:
            lr = lr * (0.1 ** (epoch // 30))
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            print("** current learning rate: {} **".format(lr))

        #save batch of images according to test_idxs and train_idxs
        if isinstance(test_ds, tuple):
            save_minibatch(net, test_ds[0], test_idxs, epoch, save=True)
        else:
            save_minibatch(net, test_ds, test_idxs, epoch, save=True)

        save_minibatch(net, train_ds, train_idxs, epoch, save=True)
    plt.savefig(model_root+net.get_info('model_configuration')['mode'] + "/grad_flow_plot.png")
    plt.show()

def pretrain_session(net, train_ds,test_ds, model_root, results_root, checkpoint = None, test_idxs=None, train_idxs=None, criterion=None,
                     START_EPOCH=0, MAX_EPOCH=50, lr=None, adjust_lr=False,batch_size=(64,16), num_workers=(8,2), shuffle=(True,False)):
    '''
    function to pretrain an autoencoder opitimizing only on input/output

    :param net: <torch.nn.Module> network to train
    :param train_ds: <tuple<torch.utils.data.Dataset>> tuple of training datasets (images, trackers not needed)
    :param test_ds:  <tuple<torch.utils.data.Dataset>> tuple of test datasets (images, trackers not needed)
    :param model_root: <str> path to where to save the model
    :param results_root: <str> path to directory to save epochs' results
    :param checkpoint: <dict> checkpoint from which to restart the training
    :param test_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of test_ds that we want to plot (with 'save_minibatch()')
    :param train_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of train_ds that we want to plot (with 'save_minibatch()')
    :param criterion1: <torch.nn.loss> criterion on image (default = None, default set later)
    :param START_EPOCH: <int> starting training epoch (default = 0)
    :param MAX_EPOCH: <int> max training epoch (default = 50)
    :param lr: <int> learning rate (default = None, so that optimizer uses its default value)
    :param adjust_lr: <bool> flag indicating if to use function adjust_learning_rate() (default = False)
    :param batch_size: <tuple<int>> tuple with batch sizes for train and test loaders (default = (64,16))
    :param num_workers: <tuple<int>> tuple with number of workers to load train and test datasets (default = (8, 2))
    :param shuffle: <tuple<bool>> tuple indicating whether to shuffle or not train and test loaders (default = (True, False))
    '''

    #check if model_root and results_root directories exist
    # (TO DO: check structure of results root, 'save_minibatch()' expects a particular directory sructure)
    assert os.path.isdir(model_root), "ERROR in pretrain_session: directory to save the model <model_root: {}> not found.".format(model_root)
    assert os.path.isdir(results_root), "ERROR in pretrain_session: directory to save results <results_root: {}> not found.".format(results_root)
    assert isinstance(train_ds, tuple), "ERROR in pretrain_session: parameter 'train_ds' must be an instance of <tuple>"
    assert isinstance(test_ds, tuple), "ERROR in pretrain_session: parameter 'test_ds' must be an instance of <tuple>"
    model_root = os.path.join(model_root, 'pretraining/')
    results_root = os.path.join(results_root, 'pretraining/')

    #determine indices of images to show at each epoch
    np.random.seed(42)
    #FIX ME: only indices of first dataset for now
    test_idxs = np.random.randint(len(test_ds[0]), size=(5,))
    train_idxs = np.random.randint(len(train_ds[0]), size=(5,))
    # create data loaders
    train_loaders = [torch.utils.data.DataLoader(ds, batch_size=batch_size[0], num_workers=num_workers[0], shuffle = shuffle[0]) for ds in train_ds]
    test_loaders = [torch.utils.data.DataLoader(ds, batch_size=batch_size[1], num_workers=num_workers[1], shuffle = shuffle[1]) for ds in test_ds]

    #instanciate optimizer
    if lr is not None:
        optim = torch.optim.Adam(net.parameters(),lr = lr)
    else:
        optim = torch.optim.Adam(net.parameters())

    #instanciate criterions if None
    if criterion is None:
        criterion = torch.nn.MSELoss()

    #move to GPU
    if use_cuda:
        net.to(device)
        for loader in train_loaders:
            loader.pin_memory = True
        for loader in test_loaders:
            loader.pin_memory = True
        cudnn.benchmark = True
        criterion.cuda

    net.update_info('model_root', model_root)
    net.update_info("results_root", results_root)
    net.update_info("epoch", START_EPOCH)
    net.update_info("best_val_loss", sys.float_info.max)
    net.update_info("best_train_loss", sys.float_info.max)
    net.update_info("val_loss", sys.float_info.max)
    net.update_info("train_loss", sys.float_info.max)
    #print info before starting the training loop
    net.print_info(verbose=True)

    # declare losses dict
    losses = {'train': [], 'validate': []}

    # loop through epochs
    for epoch in range(START_EPOCH + 1, MAX_EPOCH + 1):
        print("\n\t\t\t**EPOCH: [{}/{}]**\n".format(epoch, MAX_EPOCH))
        #train
        net.train()
        final_loss = 0
        print("\n**TRAINING.....**\n\n")
        for idx, loader in enumerate(train_loaders):
            running_loss = 0
            for batch_idx, (img, _) in enumerate(loader):
                img = img.to(device)
                img = img.type(torch.cuda.FloatTensor)
                # forward pass
                out, _ = net(img)
                loss = criterion(out, img)
                optim.zero_grad()
                loss.backward()
                optim.step()
                running_loss+=loss.item()
                # print some information about where we are in training
                if batch_idx % 10 == 0:
                    print("--> {}% of loader [{}]/[{}]\t running_loss: {:.5f}".format(batch_idx * 100 // len(loader), idx+1, len(train_loaders), running_loss))
            final_loss+=running_loss*(len(train_ds[idx])/sum([len(ds) for ds in train_ds])) / len(train_loaders[idx])
        #append average train loss
        losses['train'].append(final_loss)
        net.update_info('train_loss', final_loss)
        #validate
        net.eval()
        final_loss = 0
        print("\n**VALIDATING.....**\n\n")
        for idx, loader in enumerate(test_loaders):
            running_loss=0
            for batch_idx, (img, _) in enumerate(loader):
                img = img.to(device)
                img = img.type(torch.cuda.FloatTensor)
                # forward pass
                out, _ = net(img)
                loss = criterion(out, img)
                running_loss += loss.item()
                # print some information about where we are in training
                if batch_idx % 10 == 0:
                    print("--> {}% of loader [{}]/[{}]\t running_loss: {:.5f}".format(batch_idx * 100 // len(loader), idx+1, len(test_loaders), running_loss))
            final_loss += running_loss * (len(test_ds[idx])/sum([len(ds) for ds in test_ds])) / len(test_loaders[idx])
        #append average validate loss
        losses['validate'].append(final_loss)
        net.update_info('val_loss', final_loss)
        #save model
        save(net, losses, epoch, START_EPOCH)
        #print epoch summary
        net.training_summary()

        if adjust_lr and lr > 10e-6:
            lr = lr * (0.1 ** (epoch // 30))
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            print("** current learning rate: {} **".format(lr))

        #save batch of images according to test_idxs and train_idxs
        #FIX ME: prints only images of the first dataset given
        save_minibatch(net, test_ds[0], test_idxs, epoch, save=True)
        save_minibatch(net, train_ds[0], train_idxs, epoch, save=True)

def save_refined(net,losses, optim, epoch, start_epoch):
    '''
    function to save network being refined (freezing either encoder or decoder weights to a checkpoint,
     three copies of the network are being saved: last.tar, best_training.tar, best_validation.tar

    :param nets: <torch.nn.Module> network
    :param optim: <torch.optim> optimiser
    :param losses: <dict> containing training  losses and validation losses
    :param epoch: <int> current training epoch
    :param start_epoch: <int> start training epoch

    EXAMPLE TO CONSTRUCT INPUT LOSSES DICTIONARY:

    losses = {}
    losses['train'] = []
    losses['validate'] = []
    '''

    # save best validation loss network
    if losses['validate'][-1] < net.get_info('best_val_loss'):
        net.update_info('best_val_loss', losses['validate'][-1])
        print("\nSAVING AS BEST VALIDATION MODEL: {}".format(net.get_info('model_name')))
        savepath = net.get_info('model_root') + net.get_info('model_configuration')['mode'] + "/best_validation.tar"
        torch.save({'model_state_dict': net.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'validation_loss_hist': losses['validate'],
                    'train_loss_hist': losses['train'],
                    'model_info': net.get_info(),
                    }, savepath)
    else:
        print("\nperformed worse on validation set... not saving model {}".format(net.get_info('model_name')))
    if losses['train'][-1] < net.get_info('best_train_loss'):
        net.update_info('best_train_loss', losses['train'][-1])
        print("SAVING AS BEST TRAINING MODEL: {}".format(net.get_info('model_name')))
        savepath = net.get_info('model_root') + net.get_info('model_configuration')['mode'] + "/best_training.tar"
        torch.save({'model_state_dict': net.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'validation_loss_hist': losses['validate'],
                    'train_loss_hist': losses['train'],
                    'model_info': net.get_info(),
                    }, savepath)
    else:
        print("\nperformed worse on training set... not saving model {}".format(net.get_info('model_name')))

    print("SAVING AS LAST TRAINING MODEL: {}".format(net.get_info('model_name')))
    savepath = net.get_info('model_root') + net.get_info('model_configuration')['mode'] + "/last.tar"
    torch.save({'model_state_dict': net.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'validation_loss_hist': losses['validate'],
                'train_loss_hist': losses['train'],
                'model_info': net.get_info(),
                }, savepath)


def finetune_freezing_weights(net, freeze_info, train_ds,test_ds, model_root, results_root, test_idxs, train_idxs, k1 = 1., k2 = 1., criterion = None,
                  START_EPOCH = 0, MAX_EPOCH = 500, lr = None, adjust_lr = False,batch_size = (64,16), num_workers = (8,2),
                  shuffle = (True,False)):
    '''
    function to fully finetune, validate and save an already trained autoencoder network, freezing its encoding weights and only training the decoder part

    :param net: <torch.nn.Module> network to train

    :param freeze_info: <dict> dict structured like so: {'mode': 'encoder' or 'decoder' , 'delete_layers': list/range/tuple of layer (number) to delete}
            e.g. freeze_info = {'mode': 'encoder', 'delete_layers': range(5:15)} will freeze layers 5-14 of the encoder
                freeze_info = {'mode': 'decoder', 'delete_layers': [2,4,6,8,10] or (2,4,6,8,10)} will freeze layers 2,4,6,8,10 of the decoder
                TO FREEZE ALL LAYERS: freeze_info = {'mode': 'encoder'}

    :param train_ds: <torch.utils.data.Dataset> training dataset (images + trackers)
    :param test_ds:  <torch.utils.data.Dataset> test dataset (images + trackers)
    :param model_root: <str> path to where to save the model
    :param results_root: <str> path to directory to save epochs' results
    :param test_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of test_ds that we want to plot (with 'save_minibatch()')
    :param train_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of train_ds that we want to plot (with 'save_minibatch()')
    :param k1: <float> weight on criterion1 (default = 1)
    :param k2: <float> weight on criterion2 (default = 1)
    :param criterion: <torch.nn.loss> criterion on image (default = None, default set later)
    :param START_EPOCH: <int> starting training epoch (default = 0)
    :param MAX_EPOCH: <int> max training epoch (default = 500)
    :param lr: <int> learning rate (default = None, so that optimizer uses its default value)
    :param adjust_lr: <bool> flag indicating if to use function adjust_learning_rate() (above, default = False)
    :param batch_size: <tuple<int>> tuple with batch sizes for train and test loaders (default = (64,16))
    :param num_workers: <tuple<int>> tuple with number of workers to load train and test datasets (default = (8, 2))
    :param shuffle: <tuple<bool>> tuple indicating whether to shuffle or not train and test loaders (default = (True, False))

    code snipped to launch training:


    #REFINE from 500
    if START_FROM_CHECPOINT:
        #FIRST TRAIN ON NOT SCALED DATASET (DATASET 1)
        checkpoint = torch.load(model_root1 + "deconv/last.tar", map_location='cpu')
        net = Generator(checkpoint['model_info']['model_configuration'])
        net.load_state_dict(checkpoint['model_state_dict'])
        for key in checkpoint['model_info']:
            net.update_info(key,checkpoint['model_info'][key])
        engine.train_session_frozen_weights(net,{"mode": 'encoder'},train_ds1,test_ds1,model_root1,results_root1,
                                            test_idxs, train_idxs,lr = 0.0001, START_EPOCH=net.get_info("epoch"), MAX_EPOCH=1000)
        #TRAIN ON SCALED DATASET (DATASET 2)
        checkpoint = torch.load(model_root2 + "deconv/last.tar", map_location='cpu')
        net = Generator(checkpoint['model_info']['model_configuration'])
        net.load_state_dict(checkpoint['model_state_dict'])
        for key in checkpoint['model_info']:
            net.update_info(key,checkpoint['model_info'][key])
        engine.train_session_frozen_weights(net,{"mode": 'encoder'}, train_ds2, test_ds2, model_root2, results_root2,
                                            test_idxs, train_idxs, lr=0.0001, START_EPOCH=net.get_info("epoch"), MAX_EPOCH=1000)
    else:
        #FIRST TRAIN ON NOT SCALED DATASET (DATASET 1)
        net = Generator(cfg)
        engine.train_session(net,train_ds1,test_ds1,model_root1,results_root1, test_idxs, train_idxs,lr = 0.0001)
        #TRAIN ON SCALED DATASET (DATASET 2)
        net = Generator(cfg)
        engine.train_session(net, train_ds2, test_ds2, model_root2, results_root2, test_idxs, train_idxs, lr=0.0001)

    '''

    #check if model_root and results_root directories exist
    # (TO DO: check structure of results root, 'save_minibatch()' expects a particular directory sructure)
    assert os.path.isdir(model_root), "ERROR: directory to save the model <model_root: {}> not found.".format(model_root)
    assert os.path.isdir(results_root), "ERROR: directory to save results <results_root: {}> not found.".format(results_root)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size[0], num_workers=num_workers[0], shuffle = shuffle[0])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size[1], num_workers=num_workers[1], shuffle = shuffle[1])

    #freeze encoding weights
    assert 'mode' in freeze_info, "ERROR in train_session_frozen_weights(): missing key in 'freeze_info', key: ('mode')"
    assert freeze_info['mode'] in ('encoder', 'decoder'),"ERROR in train_session_frozen_weights(): freeze_info['mode'] must be in ('encoder','decoder')"

    # name of folder for saving the model later

    NAME = "frozen_layers__"
    if 'delete_layers' in freeze_info:
        assert isinstance(freeze_info['delete_layers'], (range,list,tuple)), "ERROR in train_session_frozen_weights(): key (delete_layers) must be an instance of: (range,list,tuple)"

        for layer in freeze_info['delete_layers']:
            NAME += "{}_".format(layer)
        # freeze specified layers of encorder or decoder
        for name, param in net.named_parameters():
            if freeze_info['mode'] in name and int(name.split('.')[1]) in freeze_info['delete_layers']:
                param.requires_grad = False
    else:
        #name of folder for saving the model later
        NAME += "all_"
        #freeze all weights of encorder or decoder
        for name, param in net.named_parameters():
            if freeze_info['mode'] in name:
                param.requires_grad = False

    #save the current info as 'old_info' (so that the new model has all the info of the old model in case we forget where it comes from)
    info = {}
    for key in net.get_info():
        info[key] = net.get_info(key)
    net.update_info('old_info', info)
    #update info for new training
    net.update_info('model_root', model_root) # + "refined_models/" + "frozen_{}/".format(freeze_info['mode']) + NAME[:-1] + "/")
    net.update_info("results_root", results_root) # + "refined_models/" + "frozen_{}/".format(freeze_info['mode']) + NAME[:-1] + "/")
    net.update_info("epoch", START_EPOCH)
    net.update_info("best_val_loss", sys.float_info.max)
    net.update_info("best_train_loss", sys.float_info.max)
    net.update_info("val_loss", sys.float_info.max)
    net.update_info("train_loss", sys.float_info.max)
    #print info before starting the training loop
    net.print_info(verbose=True)
    #instanciate optimizer
    if lr is not None:
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr = lr)
    else:
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))

    #instanciate criterions if None
    if criterion is None:
        criterion = torch.nn.MSELoss()

    #move to GPU
    if use_cuda:
        net.to(device)
        train_loader.pin_memory = True
        test_loader.pin_memory = True
        cudnn.benchmark = True
        criterion.cuda

    # declare losses dict (only one loss now)
    losses = {}
    losses['train'] = []
    losses['validate'] = []

    # loop through epochs
    for idx, epoch in enumerate(range(START_EPOCH + 1, MAX_EPOCH + 1)):

        net.update_info("epoch", epoch)
        print("\n\n\t\t\t**REFINING EPOCH: [{}/{}], STARTED REFINING FROM:{}**\n".format(epoch, MAX_EPOCH,net.get_info('old_info')['epoch']))
        print("\ntraining MODEL: {}\n\n".format(net.get_info('model_name')))
        net.train()
        running_loss = 0
        for batch_idx, (img, target) in enumerate(train_loader):
            img = img.to(device).type(tensortype)
            target = target.to(device).type(tensortype)

            # forward pass
            if freeze_info['mode'] == 'encoder':
                out = net.decode(target)
                loss = criterion(out, img)
            else:
                out = net.encode(img)
                loss = criterion(out,target)
            # backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
            # print some information about where we are in training
            if batch_idx % 10 == 0:
                print("--> {}%\trunning_loss: {:.5f}".format(batch_idx * 100 // len(train_loader), running_loss))
        # store average loss
        net.update_info('train_loss', running_loss / len(train_loader))
        losses['train'].append(running_loss/ len(train_loader))

        print("\nvalidating MODEL: {}\n\n".format(net.get_info('model_name')))
        net.eval()
        running_loss = 0
        for batch_idx, (img, target) in enumerate(test_loader):
            img = img.to(device).type(tensortype)
            target = target.to(device).type(tensortype)

            # forward pass
            if freeze_info['mode'] == 'encoder':
                out = net.decode(target)
                loss = criterion(out, img)
            else:
                out = net.encode(img)
                loss = criterion(out, target)
            running_loss += loss.item()
            # print some information about where we are in training
            if batch_idx % 10 == 0:
                print("--> {}%\trunning_loss: {:.5f}".format(batch_idx * 100 // len(test_loader), running_loss))
        # store average loss
        net.update_info('val_loss', running_loss/len(test_loader))
        losses['validate'].append(running_loss/ len(test_loader))

        save_refined(net, losses, epoch, START_EPOCH)

        print("\n**EPOCH RESULTS**")
        net.training_summary()
        if adjust_lr and lr > 10e-6:
            lr = lr * (0.1 ** (epoch // 2))
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            print("** current learning rate: {} **".format(lr))

        save_minibatch(net, test_ds, test_idxs, epoch, save=True)
        save_minibatch(net, train_ds, train_idxs, epoch, save=True)

def multistep_training_session(net, train_ds,test_ds, model_root, results_root, test_idxs, train_idxs, k1 = 1., k2 = 1., criterion1 = None,
                  criterion2 = None, START_EPOCH = 0, MAX_EPOCH = 200, lr = None, adjust_lr = False,batch_size = (64,16), num_workers = (8,2),
                  shuffle = (True,False)):
    '''
    function to train, validate and save an autoencoder network according to the following steps:
        1. freeze decoder weights and train encoder part to learn relation between image and latent vector z
        2. unfreeze all weigths and train autoencoder
        3. freeze encoder weigths and train decoder part

    :param net: <torch.nn.Module> network to train
    :param train_ds: <torch.utils.data.Dataset> training dataset (images + trackers)
    :param test_ds:  <torch.utils.data.Dataset> test dataset (images + trackers)
    :param model_root: <str> path to where to save the model
    :param results_root: <str> path to directory to save epochs' results
    :param test_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of test_ds that we want to plot (with 'save_minibatch()')
    :param train_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of train_ds that we want to plot (with 'save_minibatch()')
    :param k1: <float> weight on criterion1 (default = 1)
    :param k2: <float> weight on criterion2 (default = 1)
    :param criterion1: <torch.nn.loss> criterion on image (default = None, default set later)
    :param criterion2: <torch.nn.loss> criterion on latent vector z (default = None, default set later)
    :param START_EPOCH: <int> starting training epoch (default = 0)
    :param MAX_EPOCH: <int> max training epoch (default = 500)
    :param lr: <int> learning rate (default = None, so that optimizer uses its default value)
    :param adjust_lr: <bool> flag indicating if to use function adjust_learning_rate() (above, default = False)
    :param batch_size: <tuple<int>> tuple with batch sizes for train and test loaders (default = (64,16))
    :param num_workers: <tuple<int>> tuple with number of workers to load train and test datasets (default = (8, 2))
    :param shuffle: <tuple<bool>> tuple indicating whether to shuffle or not train and test loaders (default = (True, False))

    '''

    #check if model_root and results_root directories exist
    # (TO DO: check structure of results root, 'save_minibatch()' expects a particular directory sructure)
    assert os.path.isdir(model_root), "ERROR: directory to save the model <model_root: {}> not found.".format(model_root)
    assert os.path.isdir(results_root), "ERROR: directory to save results <results_root: {}> not found.".format(results_root)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size[0], num_workers=num_workers[0], shuffle = shuffle[0])
    if isinstance(test_ds, tuple):
        assert len(test_ds) == 2, "ERROR in train_session: currently developed for max 2 test datasets."
        test_loader = torch.utils.data.DataLoader(test_ds[0], batch_size=batch_size[1], num_workers=num_workers[1], shuffle = shuffle[1])
        test_loader1 = torch.utils.data.DataLoader(test_ds[1], batch_size=batch_size[1], num_workers=num_workers[1], shuffle = shuffle[1])
    else:
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size[1], num_workers=num_workers[1], shuffle = shuffle[1])

    #update info for training
    net.update_info('model_root', model_root)
    net.update_info("results_root", results_root)
    net.update_info("epoch", START_EPOCH)
    net.update_info("best_val_loss", sys.float_info.max)
    net.update_info("best_train_loss", sys.float_info.max)
    net.update_info("val_loss", sys.float_info.max)
    net.update_info("train_loss", sys.float_info.max)
    #print info before starting the training loop
    net.print_info(verbose=True)

    #instanciate optimizer
    if lr is not None:
        optim_enc = torch.optim.Adam([param for name, param in net.named_parameters() if 'encoder' in name], lr = lr)
        optim_dec = torch.optim.Adam([param for name, param in net.named_parameters() if 'decoder' in name], lr=lr)
        optim_auto = torch.optim.Adam(net.parameters(), lr=lr/2)
    else:
        optim_enc = torch.optim.Adam([param for name, param in net.named_parameters() if 'encoder' in name])
        optim_dec = torch.optim.Adam([param for name, param in net.named_parameters() if 'decoder' in name])
        optim_auto = torch.optim.Adam(net.parameters(), lr = 0.001/2) #0.001 is default for adam optimizer

    #instanciate criterions if None
    if criterion1 is None:
        criterion1 = torch.nn.MSELoss()
    if criterion2 is None:
        criterion2 = torch.nn.MSELoss()

    #move to GPU
    if use_cuda:
        net.to(device)
        train_loader.pin_memory = True
        test_loader.pin_memory = True
        if isinstance(test_ds, tuple):
            test_loader1.pin_memory = True
        cudnn.benchmark = True
        criterion1.cuda
        criterion2.cuda

    # declare losses dict
    losses = {}
    losses['train'] = {'encoder': [], 'decoder': [], 'autoencoder': []}
    if isinstance(test_ds, tuple):
        losses['validate'] = {'A': {'encoder': [], 'decoder': [], 'autoencoder': []},
                              'B': {'encoder': [], 'decoder': [], 'autoencoder': []}
                              }
    else:
        losses['validate'] = {'encoder': [], 'decoder': [], 'autoencoder': []}

    # loop through epochs
    for idx, epoch in enumerate(range(START_EPOCH + 1, MAX_EPOCH + 1)):
        net.update_info("epoch", epoch)
        print("\n\n\t\t\t**EPOCH: [{}/{}]**\n".format(epoch, MAX_EPOCH))
        print("\ntraining MODEL: {}\n\n".format(net.get_info('model_name')))
        net.train()
        running_loss_auto = 0
        running_loss_dec = 0
        running_loss_enc = 0
        for batch_idx, (img, target) in enumerate(train_loader):
            img = img.to(device).type(tensortype)
            target = target.to(device).type(tensortype)

            #forward pass freeze decoder
            net.freeze_weights('decoder')
            Z = net.encode(img)
            loss_enc = k2*criterion2(Z, target)
            # backward pass freeze decoder
            optim_enc.zero_grad()
            loss_enc.backward()
            optim_enc.step()
            running_loss_enc += loss_enc.item()
            #unfreeze weights
            net.unfreeze_weights()

            #forward pass autoencoder
            out, Z = net(img)
            loss_auto = k1*criterion1(out, img) + k2*criterion2(Z, target)
            # backward pass autoencoder
            optim_auto.zero_grad()
            loss_auto.backward()
            optim_auto.step()
            running_loss_auto += loss_auto.item()
            net.unfreeze_weights()

            # forward pass freeze encoder
            net.freeze_weights('encoder')
            out = net.decode(target)
            loss_dec = k1 * criterion1(out, img)
            # backward pass freeze encoder
            optim_dec.zero_grad()
            loss_dec.backward()
            optim_dec.step()
            running_loss_dec += loss_dec.item()
            net.unfreeze_weights()

            # print some information about where we are in training
            if batch_idx % 10 == 0:
                print("--> {}%\trunning_loss_dec: {:.5f}\trunning_loss_enc: {:.5f}\trunning_loss_auto: {:.5f}".format(batch_idx * 100 // len(train_loader), running_loss_dec, running_loss_enc, running_loss_auto))
        # store average losses
        net.update_info('train_loss', running_loss_dec/len(train_loader))
        losses['train']['encoder'].append(running_loss_enc/len(train_loader))
        losses['train']['decoder'].append(running_loss_dec/len(train_loader))
        losses['train']['autoencoder'].append(running_loss_auto/len(train_loader))

        print("\nvalidating MODEL: {} , {}\n\n".format(net.get_info('model_name'), "REGION: A" if isinstance(test_ds, tuple) else " "))
        net.eval()
        running_loss_auto = 0
        running_loss_dec = 0
        running_loss_enc = 0
        for batch_idx, (img, target) in enumerate(test_loader):
            img = img.to(device).type(tensortype)
            target = target.to(device).type(tensortype)

            #forward pass freeze decoder
            Z = net.encode(img)
            loss_enc = k2*criterion2(Z, target)
            running_loss_enc += loss_enc.item()

            #forward pass autoencoder
            out, Z = net(img)
            loss_auto = k1*criterion1(out, img) + k2*criterion2(Z, target)
            running_loss_auto += loss_auto.item()

            # forward pass freeze encoder
            out = net.decode(target)
            loss_dec = k1 * criterion1(out, img)
            running_loss_dec += loss_dec.item()
            # print some information about where we are in training
            if batch_idx % 10 == 0:
                print("--> {}%\trunning_loss_dec: {:.5f}\trunning_loss_enc: {:.5f}\trunning_loss_auto: {:.5f}".format(batch_idx * 100 // len(test_loader), running_loss_dec, running_loss_enc, running_loss_auto))
        # store average losses
        net.update_info('val_loss', running_loss_dec/len(test_loader))
        if isinstance(test_ds, tuple):
            losses['validate']['A']['encoder'].append(running_loss_enc/len(test_loader))
            losses['validate']['A']['decoder'].append(running_loss_dec/len(test_loader))
            losses['validate']['A']['autoencoder'].append(running_loss_auto/len(test_loader))
        else:
            losses['validate']['encoder'].append(running_loss_enc/len(test_loader))
            losses['validate']['decoder'].append(running_loss_dec/len(test_loader))
            losses['validate']['autoencoder'].append(running_loss_auto/len(test_loader))

        if isinstance(test_ds, tuple):
            print("\nvalidating MODEL: {} , REGION: B\n\n".format(net.get_info('model_name')))
            net.eval()
            running_loss_auto = 0
            running_loss_dec = 0
            running_loss_enc = 0
            for batch_idx, (img, target) in enumerate(test_loader1):
                img = img.to(device).type(tensortype)
                target = target.to(device).type(tensortype)

                # forward pass freeze decoder
                Z = net.encode(img)
                loss_enc = k2 * criterion2(Z, target)
                running_loss_enc += loss_enc.item()

                # forward pass autoencoder
                out, Z = net(img)
                loss_auto = k1 * criterion1(out, img) + k2 * criterion2(Z, target)
                running_loss_auto += loss_auto.item()

                # forward pass freeze encoder
                out = net.decode(target)
                loss_dec = k1 * criterion1(out, img)
                running_loss_dec += loss_dec.item()
                # print some information about where we are in training
                if batch_idx % 10 == 0:
                    print("--> {}%\trunning_loss_dec: {:.5f}\trunning_loss_enc: {:.5f}\trunning_loss_auto: {:.5f}".format(batch_idx * 100 // len(test_loader1), running_loss_dec, running_loss_enc, running_loss_auto))
            # store average losses
            losses['validate']['B']['encoder'].append(running_loss_enc/len(test_loader1))
            losses['validate']['B']['decoder'].append(running_loss_dec/len(test_loader1))
            losses['validate']['B']['autoencoder'].append(running_loss_auto/len(test_loader1))

        if adjust_lr and lr > 10e-6:
            lr = lr * (0.1 ** (epoch // 50))
            for param_group1,param_group2,param_group3 in zip(optim_enc.param_groups,optim_dec.param_groups,optim_auto.param_groups):
                param_group1['lr'] = lr
                param_group2['lr'] = lr
                param_group3['lr'] = lr/2
            print("** current learning rate: {} **".format(lr))

        #save model
        save(net, losses, epoch, START_EPOCH)
        #print epoch summary
        net.training_summary()

        #print batch of images
        if isinstance(test_ds, tuple):
            save_minibatch(net, test_ds[0], test_idxs, epoch, save=True)
        else:
            save_minibatch(net, test_ds, test_idxs, epoch, save=True)
        save_minibatch(net, train_ds, train_idxs, epoch, save=True)


def GAN_training(train_ds,test_ds, model_root, results_root, test_idxs, train_idxs, lr, gen_skip_update_epochs = 1, checkpoint = None,
                  START_EPOCH = 0, MAX_EPOCH = 500, adjust_lr = False,batch_size = (64,16), num_workers = (8,2),
                  shuffle = (True,False)):
    '''
    function to finetune, validate and save a previously trained generator network via GAN approach.
    given datasets and desired root directories

    :param train_ds: <torch.utils.data.Dataset> training dataset (images + trackers)
    :param test_ds:  <torch.utils.data.Dataset> test dataset (images + trackers)
    :param model_root: <str> path to where to save the model
    :param results_root: <str> path to directory to save epochs' results
    :param test_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of test_ds that we want to plot (with 'save_minibatch()')
    :param train_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of train_ds that we want to plot (with 'save_minibatch()')
    :param lr: <float> learning rate, so that optimizer uses its default value)
    :param gen_skip_update_epochs: int , number of epoch to skip each time to update the generator
    :param checkpoint <dictionary> dict containing a torch checkpoint to resume training from
    :param START_EPOCH: <int> starting training epoch (default = 0)
    :param MAX_EPOCH: <int> max training epoch (default = 500)
    :param adjust_lr: <bool> flag indicating if to use function adjust_learning_rate() (above, default = False)
    :param batch_size: <tuple<int>> tuple with batch sizes for train and test loaders (default = (64,16))
    :param num_workers: <tuple<int>> tuple with number of workers to load train and test datasets (default = (8, 2))
    :param shuffle: <tuple<bool>> tuple indicating whether to shuffle or not train and test loaders (default = (True, False))

    '''

    # check if model_root and results_root directories exist
    assert os.path.isdir(model_root), "ERROR: directory to save the model <model_root: {}> not found.".format(
        model_root)
    assert os.path.isdir(results_root), "ERROR: directory to save results <results_root: {}> not found.".format(
        results_root)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size[0], num_workers=num_workers[0],
                                               shuffle=shuffle[0])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size[1], num_workers=num_workers[1],
                                              shuffle=shuffle[1])

    #instanciate networks
    from architectures.Discriminator import Conditional_Discriminator
    from architectures.Discriminator import  Latent_Discriminator
    from architectures.GAN_generators import Conditional_Adversarial_Autoencoder

    netG = Conditional_Adversarial_Autoencoder(z_size=7,noise_size=100, latent_image_size=(32, 8, 8))
    netD = Conditional_Discriminator(z_size=7, latent_image_size=(32, 8, 8))
    netD_tracker = Latent_Discriminator(z_size = 7)
    netD_noise = Latent_Discriminator(z_size = 100)

    # custom weights initialization if needed
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)


    #instanciate optimizers
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr*1.5, betas=(0.5, 0.999))
    optimizerD_z = torch.optim.Adam(netD_tracker.parameters(), lr=lr*1.5, betas=(0.5, 0.999))
    optimizerD_noise = torch.optim.Adam(netD_noise.parameters(), lr=lr*1.5, betas=(0.5, 0.999))
    optimizerG_decoder = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG_encoder = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_tracker = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


    if checkpoint is None:

        netD.apply(weights_init)
        netG.apply(weights_init)
        netD_tracker.apply(weights_init)
        netD_noise.apply(weights_init)

        netG.update_info('model_root', model_root)
        netG.update_info("results_root", results_root)
        netG.update_info("epoch", START_EPOCH)

        #print info before starting the training loop
        netG.print_info(verbose=True)

        G_losses, D_losses, tracker_losses = [],[],[]

    else:
        #TODO!!!!!!!!!!!!!!!!
        netG.load_state_dict(checkpoint['netG_state_dict'])
        G_losses = checkpoint['loss_G']
        D_losses = checkpoint['loss_D']
        netG.print_info(verbose=True)

    #move to GPU
    if use_cuda:
        netG.to(device)
        netD.to(device)
        netD_noise.to(device)
        netD_tracker.to(device)
        train_loader.pin_memory = True
        test_loader.pin_memory = True
        cudnn.benchmark = True

    TINY = 1e-15
    criterion = torch.nn.MSELoss()
    print("Starting Training Loop...")
    # For each epoch
    for idx, epoch in enumerate(range(START_EPOCH + 1, MAX_EPOCH + 1)):
        netG.update_info("epoch", epoch)
        # TRAINING
        netD.train()
        netD_tracker.train()
        netD_noise.train()
        netG.train()
        for i, data in enumerate(train_loader, 0):

            # Format batch
            b_size = batch_size[0] if i < len(train_ds) // batch_size[0] else len(train_ds) % batch_size[0]
            # Forward pass real and fake images through D
            images, trackers = data[0].to(device).type(tensortype), data[1].to(device).type(tensortype)

            ############################
            # (1) Update D network: minimize -[log(D(images)) + log(1 - D(fakes))]
            ###########################

            netD.zero_grad()
            netD_tracker.zero_grad()
            netD_noise.zero_grad()
            netG.zero_grad()

            fakes, _, _ = netG(images)

            D_real = netD(images, trackers)
            D_fake = netD(fakes, trackers)

            # Calculate D loss
            D_loss = -torch.mean(torch.log(D_real + TINY) + torch.log(1 - D_fake + TINY))

            #backward pass
            D_loss.backward()
            optimizerD.step()

            ############################
            # (2) Update G network: minimize -[log(D(fakes))]
            ###########################

            netD.zero_grad()
            netD_tracker.zero_grad()
            netD_noise.zero_grad()
            netG.zero_grad()

            fakes, _, _ = netG(images)

            D_fake = netD(fakes, trackers)
            G_loss = -torch.mean(torch.log(D_fake + TINY))

            #backward pass
            G_loss.backward()
            optimizerG_decoder.step()

            ############################
            # (3) Update D_tracker and D_noise networks: minimize -[log(D(z)) + log(1 - D(z_fake))]
            #                                            minimize     -[log(D(noise)) + log(1 - D(noise_fake))]
            ###########################

            netD.zero_grad()
            netD_tracker.zero_grad()
            netD_noise.zero_grad()
            netG.zero_grad()

            noise = torch.randn(b_size, 100, device=device).type(tensortype)
            z = torch.randn(b_size, 7, device=device).type(tensortype)

            _, z_fake, noise_fake = netG(images)
            D_z_real, D_z_fake = netD_tracker(z), netD_tracker(z_fake)
            D_noise_real, D_noise_fake = netD_noise(noise), netD_noise(noise_fake)

            D_loss_tracker = -torch.mean(torch.log(D_z_real + TINY) + torch.log(1 - D_z_fake + TINY))
            D_loss_noise = -torch.mean(torch.log(D_noise_real + TINY) + torch.log(1 - D_noise_fake + TINY))
            D_loss_latent = D_loss_tracker + D_loss_noise

            #backward pass
            D_loss_latent.backward()
            optimizerD_z.step()
            optimizerD_noise.step()

            ############################
            # (4) Update G_tracker and G_noise networks: minimize -[log(D(z_fake))]
            #                                            minimize   -[log(D(noise_fake))]
            ###########################

            netD.zero_grad()
            netD_tracker.zero_grad()
            netD_noise.zero_grad()
            netG.zero_grad()

            _, z_fake, noise_fake = netG(images)
            D_z_fake, D_noise_fake = netD_tracker(z_fake), netD_noise(noise_fake)
            G_loss_latent = -torch.mean(torch.log(D_z_fake + TINY)) -torch.mean(torch.log(D_noise_fake + TINY))

            #backward pass
            G_loss_latent.backward()
            optimizerG_encoder.step()

            ############################
            # (5) supervised step, disentangle z on tracker with MSE
            ###########################

            netD.zero_grad()
            netD_tracker.zero_grad()
            netD_noise.zero_grad()
            netG.zero_grad()

            _, z_fake, _ = netG(images)
            loss_tracker = criterion(z_fake,trackers)

            #backward pass
            loss_tracker.backward()
            optimizer_tracker.step()

            netD.zero_grad()
            netD_tracker.zero_grad()
            netD_noise.zero_grad()
            netG.zero_grad()


            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\nD_real: %.4f\tD_fake: %.4f\n'
                      'D_z_real: %.4f\tD_z_fake: %.4f\n'
                      'D_noise_real: %.4f\tD_noise_fake: %.4f\n'
                      'MSE z tracker: %.4f'
                      % (epoch, MAX_EPOCH, i, len(train_loader),
                         D_real.mean().item(), D_fake.mean().item(), D_z_real.mean().item(), D_z_fake.mean().item(),
                         D_noise_real.mean().item(), D_noise_fake.mean().item(), loss_tracker.item()))

            # Save Losses for plotting later
            G_losses.append(D_fake.mean().item())
            D_losses.append(D_real.mean().item())
            tracker_losses.append(loss_tracker.item)

        # generate images
        save_minibatch(netG, train_ds, train_idxs, epoch, save=True)
        save_minibatch(netG, test_ds, test_idxs, epoch, save=True)

        # save netG
        savepath = netG.get_info('model_root') + netG.get_info('model_configuration')['mode'] + "/last.tar"
        torch.save({'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'netD_tracker_state_dict': netD_tracker.state_dict(),
                    'netD_noise_state_dict': netD_noise.state_dict(),
                    'optimizerG_decoder_state_dict': optimizerG_decoder.state_dict(),
                    'optimizerG_encoder_state_dict': optimizerG_encoder.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'optimizerD_z_state_dict': optimizerD_z.state_dict(),
                    'optimizerD_noise_state_dict': optimizerD_noise.state_dict(),
                    'optimizer_tracker': optimizer_tracker.state_dict(),
                    'loss_G': G_losses,
                    'loss_D': D_losses,
                    'loss_tracker': tracker_losses,
                    'model_info': netG.get_info(),
                    }, savepath)


