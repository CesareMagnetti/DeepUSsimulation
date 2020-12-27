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

    losses = {'autoencoder': [], 'encoder': [], 'decoder':[], 'KLD': []}

    print("\ntraining MODEL: {}\n\n".format(net.get_info('model_name')))
    net.update_info("epoch", epoch)
    net.train()
    running_loss = 0
    running_loss_enc = 0
    running_loss_dec = 0
    running_loss_auto = 0
    running_loss_KLD = 0

    for batch_idx, (img, target) in enumerate(loader):
        img = img.to(device).type(tensortype)
        target = target.to(device).type(tensortype)

        # forward pass
        out1, z, mu, logvar = net(img)
        out2 = net.decode(target)
        loss_auto = k1 * criterion1(out1, img)
        loss_dec = k1 * criterion1(out2, img)
        loss_enc = k2 * criterion2(z, target)
        loss_KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        loss_KLD = loss_KLD.mean(1).mean(0)
        loss_KLD.mul_(net.get_info("BETA")) #multiply KLD by beta factor
        #loss_KLD = 0.5*torch.sum(torch.exp(logvar)+mu.pow(2) -1. - logvar)

        loss = loss_auto + loss_enc + loss_KLD #supervised, enforcing Z to resemble the tracking information
        loss_id = "loss_auto + loss_enc + loss_KLD"
        #loss = loss_auto + loss_KLD #unsupervised, no label constrained, normal VAE
        #loss_id = "loss_auto + loss_KLD"
        # backward pass
        optim.zero_grad()
        loss.backward()
        plot_grad_flow(net.named_parameters())
        optim.step()
        running_loss += loss.item()
        running_loss_auto += loss_auto.item()
        running_loss_dec += loss_dec.item()
        running_loss_enc += loss_enc.item()
        running_loss_KLD += loss_KLD.item()
        # print some information about where we are in training
        if batch_idx % 10 == 0:
            print("\n--> {}%\trunning_loss_dec: {:.5f}\trunning_loss_enc: {:.5f}\trunning_loss_auto: {:.5f}\t"\
                  "running_loss_KLD: {:.5f}".format(batch_idx * 100 // len(loader), running_loss_dec, running_loss_enc, running_loss_auto, running_loss_KLD))
            print("    \ttotal loss under optimization: {} = {:.5f}".format(loss_id, running_loss))
    # store average loss
    net.update_info('train_loss', running_loss/ len(loader))
    losses['autoencoder'].append(running_loss_auto/ len(loader))
    losses['decoder'].append(running_loss_dec/ len(loader))
    losses['encoder'].append(running_loss_enc/ len(loader))
    losses['KLD'].append(running_loss_KLD / len(loader))
    return losses


# validate
def validate_epoch(net, loader, epoch, criterion1, criterion2, k1, k2, update_loss = True):
    '''
    function to validate a variational autoencoder networks (and if needed optoimizers) over a given testing loader
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

    losses = {'autoencoder': [], 'encoder': [], 'decoder': [], 'KLD': []}

    print("\nvalidating MODEL: {}\n\n".format(net.get_info('model_name')))
    net.update_info("epoch", epoch)
    net.eval()
    running_loss = 0
    running_loss_auto = 0
    running_loss_dec = 0
    running_loss_enc = 0
    running_loss_KLD = 0
    for batch_idx, (img, target) in enumerate(loader):
        img = img.to(device).type(tensortype)
        target = target.to(device).type(tensortype)

        # forward pass
        out1, z, mu, logvar = net(img)
        out2 = net.decode(target)
        loss_auto = k1 * criterion1(out1, img)
        loss_dec = k1 * criterion1(out2, img)
        loss_enc = k2 * criterion2(z, target)
        loss_KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        loss_KLD = loss_KLD.mean(1).mean(0)
        loss_KLD.mul_(net.get_info("BETA")) #multiply KLD by beta factor
        #loss_KLD = 0.5*torch.sum(torch.exp(logvar)+mu.pow(2) -1. - logvar)
        loss = loss_auto + loss_enc + loss_KLD #supervised, enforcing Z to resemble the tracking information
        loss_id = "loss_auto + loss_enc + loss_KLD"
        #loss = loss_auto + loss_KLD #unsupervised, no label constrained, normal VAE
        #loss_id = "loss_auto + loss_KLD"
        running_loss += loss.item()
        running_loss_auto += loss_auto.item()
        running_loss_dec += loss_dec.item()
        running_loss_enc += loss_enc.item()
        running_loss_KLD += loss_KLD.item()
        # print some information about where we are in training
        if batch_idx % 10 == 0:
            print("\n--> {}%\trunning_loss_dec: {:.5f}\trunning_loss_enc: {:.5f}\trunning_loss_auto: {:.5f}\t"\
                  "running_loss_KLD: {:.5f}".format(batch_idx * 100 // len(loader), running_loss_dec, running_loss_enc, running_loss_auto, running_loss_KLD))
            print("    \ttotal loss under optimization: {} = {:.5f}".format(loss_id, running_loss))
    #update model loss if needed
    if update_loss:
        net.update_info('val_loss', running_loss/ len(loader))
    # store average loss
    losses['autoencoder'].append(running_loss_auto/ len(loader))
    losses['decoder'].append(running_loss_dec/ len(loader))
    losses['encoder'].append(running_loss_enc/ len(loader))
    losses['KLD'].append(running_loss_KLD / len(loader))
    return losses



def save(net,optim,losses):
    '''
    function to save network to a checkpoint, three copies of the network are being saved:
    last.tar, best_training.tar, best_validation.tar

    :param net: <torch.nn.Module> network
    :param losses: <dict> containing training  losses and validation losses
    :param epoch: <int> current training epoch
    :param start_epoch: <int> start training epoch

    EXAMPLE TO CONSTRUCT INPUT LOSSES DICTIONARY:

    losses = {}

    losses['train'] = {'encoder': [], 'decoder': [], 'autoencoder': [], 'KLD': []}

    if isinstance(test_ds, tuple):
        losses['validate'] = {'A': {'encoder': [], 'decoder': [], 'autoencoder': [], 'KLD': []},
                              'B': {'encoder': [], 'decoder': [], 'autoencoder': [], 'KLD': []}
                              }
    '''

    # save last epoch
    print("SAVING AS LAST TRAINING MODEL: {}".format(net.get_info('model_name')))
    savepath = net.get_info('model_root') + net.get_info('model_configuration')['mode'] + "/last.tar"
    torch.save({'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
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
                    'optimizer_state_dict': optim.state_dict(),
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
                    'optimizer_state_dict': optim.state_dict(),
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
        img = img.unsqueeze(0)
        target = torch.tensor(target).unsqueeze(0)
        img = img.to(device).type(tensortype)
        target = target.to(device).type(tensortype)

        #autoencoded image
        out,_,_,_ = net(img)
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
                  START_EPOCH=0, MAX_EPOCH=200, lr=None, adjust_lr=False,batch_size=(64,16), num_workers=(8,2), shuffle=(True,False)):
    '''
    function to fully train, validate and save a variational autoencoder network (along as some epoch results) given datasets and desired root directories

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

    net.to(device)
    #instanciate optimizer
    if lr is not None:
        optim = torch.optim.Adam(net.parameters(),lr = lr)
    else:
        optim = torch.optim.Adam(net.parameters())

    # declare losses dict
    losses = {}

    if checkpoint is None:
        # setup training variables
        net.update_info('model_root', model_root)
        net.update_info("results_root", results_root)
        net.update_info("epoch", START_EPOCH)
        net.update_info("best_val_loss", sys.float_info.max)
        net.update_info("best_train_loss", sys.float_info.max)
        net.update_info("val_loss", sys.float_info.max)
        net.update_info("train_loss", sys.float_info.max)
        net.update_info("k1", k1)
        net.update_info("k2", k2)

        # print info before starting the training loop
        net.print_info(verbose=True)

        losses['train'] = {'encoder': [], 'decoder': [], 'autoencoder': [], 'KLD': []}
        if isinstance(test_ds, tuple):
            losses['validate'] = {'A': {'encoder': [], 'decoder': [], 'autoencoder': [], 'KLD': []},
                                  'B': {'encoder': [], 'decoder': [], 'autoencoder': [], 'KLD': []}
                                  }
        else:
            losses['validate'] = {'encoder': [], 'decoder': [], 'autoencoder': [], 'KLD': []}
    else:
        # restart from checkpoint
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device)
        optim.load_state_dict(checkpoint['optim_state_dict'])
        for key in checkpoint['model_info']:
            net.update_info(key, checkpoint['model_info'][key])
        losses['train'] = checkpoint['train_loss_hist']
        losses['validate'] = checkpoint['validation_loss_hist']

        # print info before starting the training loop
        net.print_info(verbose=True)

    # move to GPU
    if use_cuda:
        train_loader.pin_memory = True
        test_loader.pin_memory = True
        if isinstance(test_ds, tuple):
            test_loader1.pin_memory = True
        cudnn.benchmark = True
        criterion1.cuda
        criterion2.cuda
    # loop through epochs
    for idx, epoch in enumerate(range(START_EPOCH + 1, MAX_EPOCH + 1)):
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
        save(net, optim, losses)
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