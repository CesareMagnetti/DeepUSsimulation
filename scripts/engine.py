import torch
import torchvision.utils as vutils
import matplotlib
from matplotlib import pyplot as plt
from plotting_functions import plot_results
import numpy as np
import transforms.tensor_transforms as tensortransforms

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#function to adjust learning rate:
def adjust_learning_rate(optimizer,lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# train
def train_epoch(nets, optims, loader, epoch, criterion1, criterion2, k1=1., k2=0.1):
    '''
    function to train a tuple of networks (and if needed optoimizers) over a given training loader
    args:
      -nets <tuple>: tuple containing networks to train
      -optims <tuple>: tuple containing corresponding optimizers for each network
      -loader <torch.utils.data.dataloader>: data loader object containing LABELLED data
      -epoch <int>: training epoch the nets currently are
      -criterion1 <torch.nn>: criterion on autoencoded image
      -criterion2 <torch.nn>: criterion on latent space
      -k1 <float>: weight of criterion 1 (default 1.)
      -k2 <float>: weight of criterion 2 (default 0.)
    '''
    # error handling
    assert len(nets) == len(optims), "ERROR: tuples <nets> and <optims> must have the same length!"
    losses = {}
    losses['img'], losses['z'] = {'from_img': [], 'from_z': []}, []
    for idx, (net, optim) in enumerate(zip(nets, optims)):
        print("training MODEL: {}".format(net.get_info('model_name')))
        net.update_info("epoch", epoch)
        net.train()
        running_loss = 0
        running_loss_img1 = 0
        running_loss_img2 = 0
        running_loss_z = 0

        for batch_idx, (img, target) in enumerate(loader):
            img = img.to(device)
            target = target.to(device)
            img = img.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.FloatTensor)
            # forward pass
            out1, z = net(img)
            out2 = net.decode(target)
            loss_img1 = k1 * criterion1(out1, img)
            loss_img2 = k1 * criterion1(out2, img)
            loss_z = k2 * criterion2(z, target)
            loss = loss_img1 + loss_z
            # backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
            running_loss_img1 += loss_img1.item()
            running_loss_img2 += loss_img2.item()
            running_loss_z += loss_z.item()
            # print some information about where we are in training
            if batch_idx % 10 == 0:
                print("--> {}%".format(batch_idx * 100 // len(loader)))
        # store average loss
        net.update_info('train_loss', running_loss / len(loader))
        losses['img']['from_img'].append(running_loss_img1 / len(loader))
        losses['img']['from_z'].append(running_loss_img2 / len(loader))
        losses['z'].append(running_loss_z / len(loader))

    return losses


# validate
def validate_epoch(nets, optims, loader, epoch, criterion1, criterion2, k1=1., k2=0.1):
    '''
    function to validate a tuple of networks (and if needed optoimizers) over a given testing loader
    args:
      -nets <tuple>: tuple containing networks to validate
      -optims <tuple>: tuple containing corresponding optimizers for each network
      -loader <torch.dataloader>: data loader object containing LABELLED data
      -epoch <int>: training epoch the nets currently are
      -criterion1 <torch.nn>: criterion on autoencoded image
      -criterion2 <torch.nn>: criterion on latent space
      -k1 <float>: weight of criterion 1 (default 1.)
      -k2 <float>: weight of criterion 2 (default 0.)
    '''

    # error handling
    assert len(nets) == len(optims), "ERROR: tuples <nets> and <optims> must have the same length!"
    losses = {}
    losses['img'],losses['z'] = {'from_img': [], 'from_z': []},[]
    for idx, (net, optim) in enumerate(zip(nets, optims)):
        print("validating MODEL: {}".format(net.get_info('model_name')))
        net.update_info("epoch", epoch)
        net.eval()
        running_loss = 0
        running_loss_img1 = 0
        running_loss_img2 = 0
        running_loss_z = 0
        for batch_idx, (img, target) in enumerate(loader):
            img = img.to(device)
            target = target.to(device)
            img = img.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.FloatTensor)
            # forward pass
            out1, z = net(img)
            out2 = net.decode(target)
            loss_img1 = k1 * criterion1(out1, img)
            loss_img2 = k1 * criterion1(out2, img)
            loss_z = k2 * criterion2(z, target)
            loss = loss_img1 + loss_z
            running_loss += loss.item()
            running_loss_img1 += loss_img1.item()
            running_loss_img2 += loss_img2.item()
            running_loss_z += loss_z.item()
            # print some information about where we are in training
            if batch_idx % 10 == 0:
                print("--> {}%".format(batch_idx * 100 // len(loader)))
        # store average loss
        net.update_info('val_loss', running_loss / len(loader))
        losses['img']['from_img'].append(running_loss_img1 / len(loader))
        losses['img']['from_z'].append(running_loss_img2 / len(loader))
        losses['z'].append(running_loss_z / len(loader))
    return losses


def store_losses(train_loss, validate_loss, losses, idx):

    '''
    function to assemble the losses in a input dictionary

    EXAMPLE TO CONSTRUCT INPUT LOSSES DICTIONARY:

    losses = {}
    losses['train'] = {'img': {'from_img':np.zeros((len(nets),MAX_EPOCH-START_EPOCH)),
                                  'from_z':np.zeros((len(nets),MAX_EPOCH-START_EPOCH))},
                       'z': np.zeros((len(nets),MAX_EPOCH-START_EPOCH))}
    losses['validate'] = {'img': {'from_img':np.zeros((len(nets),MAX_EPOCH-START_EPOCH)),
                                  'from_z':np.zeros((len(nets),MAX_EPOCH-START_EPOCH))},
                          'z': np.zeros((len(nets),MAX_EPOCH-START_EPOCH))}

    :param train_loss: dict containg train losses evaluated by the train_epoch() function (in this file)
    :param validate_loss: dict containg train losses evaluated by the train_epoch() function (in this file)
    :param losses: input losses dictionary (see above example)
    :param idx: index relating to the network currently being trained (if train_epoch() and validate_epoch()
                are used to train tuples of networks)
    '''
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



def save(nets,losses, epoch, start_epoch):
    '''

    :param nets: <tuple> containing the networks being trained
    :param losses: <dict> containing training total losses and validation losses for each network
    :param epoch: <int> current training epoch
    :param start_epoch: <int> start training epoch
    '''
    train_loss_hist,val_loss_hist = {'img': {'from_img': None,'from_z': None}, 'z': None},{'img': {'from_img': None,'from_z': None}, 'z': None}
    for net, tl_img1, vl_img1,tl_img2, vl_img2, tl_z, vl_z in zip(nets, losses['train']['img']['from_img'], losses['validate']['img']['from_img'],
                                                               losses['train']['img']['from_z'],losses['validate']['img']['from_z'],
                                                               losses['train']['z'], losses['validate']['z']):

        train_loss_hist['img']['from_img'] = tl_img1[:epoch - start_epoch]
        train_loss_hist['img']['from_z'] = tl_img2[:epoch - start_epoch]
        train_loss_hist['z'] = tl_z[:epoch - start_epoch]

        val_loss_hist['img']['from_img'] = vl_img1[:epoch - start_epoch]
        val_loss_hist['img']['from_z'] = vl_img2[:epoch - start_epoch]
        val_loss_hist['z'] = vl_z[:epoch - start_epoch]

        # save best validation loss network
        if val_loss_hist['img']['from_z'][-1] + val_loss_hist['z'][-1] < net.get_info('best_val_loss'):
            net.update_info('best_val_loss', val_loss_hist['img']['from_z'][-1] + val_loss_hist['z'][-1])
            print("\nSAVING AS BEST VALIDATION MODEL: {}".format(net.get_info('model_name')))
            savepath = net.get_info('model_root') + net.get_info('model_configuration')['mode'] + "/best_validation.tar"
            torch.save({'model_state_dict': net.state_dict(),
                        'validation_loss_hist': val_loss_hist,
                        'train_loss_hist': train_loss_hist,
                        'model_info': net.get_info(),
                        }, savepath)
        else:
            print("\nperformed worse on validation set... not saving model {}".format(net.get_info('model_name')))
        if train_loss_hist['img']['from_z'][-1] + train_loss_hist['z'][-1] < net.get_info('best_train_loss'):
            net.update_info('best_train_loss', train_loss_hist['img']['from_z'][-1] + train_loss_hist['z'][-1])
            print("SAVING AS BEST TRAINING MODEL: {}".format(net.get_info('model_name')))
            savepath = net.get_info('model_root') + net.get_info('model_configuration')['mode'] + "/best_training.tar"
            torch.save({'model_state_dict': net.state_dict(),
                        'validation_loss_hist': val_loss_hist,
                        'train_loss_hist': train_loss_hist,
                        'model_info': net.get_info(),
                        }, savepath)
        else:
            print("\nperformed worse on training set... not saving model {}".format(net.get_info('model_name')))

        print("SAVING AS LAST TRAINING MODEL: {}".format(net.get_info('model_name')))
        savepath = net.get_info('model_root') + net.get_info('model_configuration')['mode'] + "/last.tar"
        torch.save({'model_state_dict': net.state_dict(),
                    'validation_loss_hist': val_loss_hist,
                    'train_loss_hist': train_loss_hist,
                    'model_info': net.get_info(),
                    }, savepath)


def save_and_display_image(nets, loader, epoch=None, plot = False, save = False):
    '''
    function to save and/or display results of multiple network on a random image of the loader
    :param nets: <tuple> containing the networks for which to evaluate the reconstructed image
    :param loader: <torch.utils.dataloader> dataloader containing images
    :param epoch (optional): <int> current training epoch
    :param plot: (optional): <bool> flag indicating if plotting the result or only saving it
    :param save: (optional): <bool> flag indicating if saving the image
    '''
    img, target = next(iter(loader))
    idx = np.random.randint(32)
    img, target = img[idx].unsqueeze(0), target[idx].unsqueeze(0)
    img = img.to(device)
    target = target.to(device)
    img = img.type(torch.cuda.FloatTensor)
    target = target.type(torch.cuda.FloatTensor)
    outs = ()
    names = ()
    for net in nets:
        name = net.get_info('model_name')
        names += (name,)
        if epoch is not None:
            if epoch < 10:
                savepath = net.get_info('results_root') + net.get_info('model_configuration')[
                    'mode'] + "/" + "EPOCH_00{}.png".format(epoch)
            elif epoch >= 10 and epoch < 100:
                savepath = net.get_info('results_root') + net.get_info('model_configuration')[
                    'mode'] + "/" + "EPOCH_0{}.png".format(epoch)
            else:
                savepath = net.get_info('results_root') + net.get_info('model_configuration')[
                    'mode'] + "/" + "EPOCH_{}.png".format(epoch)
        else:
            savepath = net.get_info('results_root') + net.get_info('model_configuration')[
                'mode'] + "/" + "TEST_IMAGE.png".format(epoch)
        out, _ = net(img)
        out = out.detach().cpu().numpy().squeeze()
        if save:
            # save image to drive
            matplotlib.pyplot.imsave(savepath, out, cmap='viridis')
        outs += (out,)
    if plot:
        plot_results(img, outs, epoch, names)


def save_and_display_minibatch(nets, ds, idxs, epoch=None, plot = False, save = False):
    '''
    function to save and/or display results of multiple network on a random image of the loader
    :param nets: <tuple> containing the networks for which to evaluate the reconstructed image
    :param ds: <torch.utils.dataset> dataset containing images
    :param epoch (optional): <int> current training epoch
    :param plot: (optional): <bool> flag indicating if plotting the result or only saving it
    :param save: (optional): <bool> flag indicating if saving the image
    '''

    outs = ()
    names = ()
    for net in nets:
        name = net.get_info('model_name')
        names += (name,)
        if epoch is not None:
            if epoch < 10:
                savepath = net.get_info('results_root') + net.get_info('model_configuration')[
                    'mode'] + "/" + "EPOCH_00{}.png".format(epoch)
            elif epoch >= 10 and epoch < 100:
                savepath = net.get_info('results_root') + net.get_info('model_configuration')[
                    'mode'] + "/" + "EPOCH_0{}.png".format(epoch)
            else:
                savepath = net.get_info('results_root') + net.get_info('model_configuration')[
                    'mode'] + "/" + "EPOCH_{}.png".format(epoch)
        else:
            savepath = net.get_info('results_root') + net.get_info('model_configuration')[
                'mode'] + "/" + "TEST_IMAGE.png".format(epoch)
        imgs,outs = [],[]
        for idx in idxs:
            img, target = ds[idx]
            img = img.detach().cpu().numpy().squeeze()
            target = torch.tensor(target).unsqueeze(0).to(device)
            target = target.type(torch.cuda.FloatTensor)
            out = net.decode(target)
            out = out.detach().cpu().numpy().squeeze()
            outs.append(out)
            imgs.append(img)

        imgs,outs = np.array(imgs),np.array(outs)
        IM = np.hstack(np.array(imgs))
        OUT = np.hstack(np.array(outs))
        IMAGE = np.vstack(np.array([IM,OUT]))

        if save:
            # save image to drive
            cmap = plt.cm.gray
            norm = plt.Normalize(vmin=0, vmax=1)
            plt.imsave(savepath, cmap(norm(IMAGE)))
        outs += (out,)

    if plot:
        plot_results(img, outs, epoch, names)

def save_and_display_minibatch2(nets, ds, idxs, epoch=None, plot=False, save=False):
    '''
    function to save and/or display results of multiple network on a random image of the loader
    :param nets: <tuple> containing the networks for which to evaluate the reconstructed image
    :param ds: <torch.utils.dataset> dataset containing images
    :param epoch (optional): <int> current training epoch
    :param plot: (optional): <bool> flag indicating if plotting the result or only saving it
    :param save: (optional): <bool> flag indicating if saving the image
    '''

    names = ()
    for net in nets:
        name = net.get_info('model_name')
        names += (name,)
        if epoch is not None:
            if epoch < 10:
                savepath = net.get_info('results_root') + net.get_info('model_configuration')[
                    'mode'] + "/" + "EPOCH_00{}.png".format(epoch)
            elif epoch >= 10 and epoch < 100:
                savepath = net.get_info('results_root') + net.get_info('model_configuration')[
                    'mode'] + "/" + "EPOCH_0{}.png".format(epoch)
            else:
                savepath = net.get_info('results_root') + net.get_info('model_configuration')[
                    'mode'] + "/" + "EPOCH_{}.png".format(epoch)
        else:
            savepath = net.get_info('results_root') + net.get_info('model_configuration')[
                'mode'] + "/" + "TEST_IMAGE.png".format(epoch)
        imgs, outs, outs1 = [], [], []
        for idx in idxs:
            img, target = ds[idx]
            img = img.unsqueeze(0).to(device)
            img = img.type(torch.cuda.FloatTensor)
            target = torch.tensor(target).unsqueeze(0).to(device)
            target = target.type(torch.cuda.FloatTensor)
            #autoencoded image
            out,_ = net(img)
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

    if plot:
        plot_results(img, outs, epoch, names)

def save_minibatch(nets, ds, idxs, epoch=None, save=False):
    '''
    function to save results of multiple network on a random minibtch of the loader
    :param nets: <tuple> containing the networks for which to evaluate the reconstructed image
    :param ds: <torch.utils.dataset> dataset containing images
    :param idxs: <np.array> array of indexes to sample from ds
    :param epoch (optional): <int> current training epoch
    :param save: (optional): <bool> flag indicating if saving the image
    '''

    names = ()
    for net in nets:
        name = net.get_info('model_name')
        names += (name,)
        if epoch is not None:
            if epoch < 10:
                savepath = net.get_info('results_root')+ ds.get_mode() + "/" +  net.get_info('model_configuration')[
                    'mode'] + "/" + "EPOCH_00{}.png".format(epoch)
            elif epoch >= 10 and epoch < 100:
                savepath = net.get_info('results_root')+ ds.get_mode() + "/" + net.get_info('model_configuration')[
                    'mode'] + "/" + "EPOCH_0{}.png".format(epoch)
            else:
                savepath = net.get_info('results_root') + ds.get_mode() + "/" + net.get_info('model_configuration')[
                    'mode'] + "/" + "EPOCH_{}.png".format(epoch)
        else:
            savepath = net.get_info('results_root')+ ds.get_mode() + "/" + net.get_info('model_configuration')[
                'mode'] + "/" + "TEST_IMAGE.png".format(epoch)
        imgs, outs, outs1 = [], [], []
        for idx in idxs:
            img, target = ds[idx]
            img = img.unsqueeze(0).to(device)
            img = img.type(torch.cuda.FloatTensor)
            target = torch.tensor(target).unsqueeze(0).to(device)
            target = target.type(torch.cuda.FloatTensor)
            #autoencoded image
            out,_ = net(img)
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