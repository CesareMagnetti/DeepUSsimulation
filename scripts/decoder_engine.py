import torch
import os
import matplotlib
from matplotlib import pyplot as plt
from plotting_functions import plot_results
import numpy as np
from torch.backends import cudnn
import sys

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
tensortype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# train
def train_epoch(net, optim, loader, epoch, criterion, minimize_criterion = True):
    '''
    function to train a tuple of networks (and if needed optoimizers) over a given training loader
    args:
      -net <torch.nn.Module>: neural network to train
      -optim <torch.optim>:  optimizer for neural network
      -loader <torch.utils.data.dataloader>: data loader object containing LABELLED data
      -epoch <int>: training epoch the nets currently are
      -criterion <torch.nn>: criterion between decoded image and original image
      -minimize_criterion: <bool> if False loss = 1 - criterion(out,input), maximizes criterion (default = True)

    '''

    print("training MODEL: {}".format(net.get_info('model_name')))
    net.update_info("epoch", epoch)
    net.train()
    running_loss = 0

    for batch_idx, (img, target) in enumerate(loader):
        img = img.to(device).type(tensortype)
        target = target.to(device).type(tensortype)

        # forward pass
        out = net(target)
        if minimize_criterion:
            loss = criterion(out,img)
        else:
            loss = - criterion(out,img)
        # backward pass
        optim.zero_grad()
        loss.backward()
        plot_grad_flow(net.named_parameters())
        optim.step()
        running_loss += loss.item()
        # print some information about where we are in training
        if batch_idx % 10 == 0:
            print("--> {}%".format(batch_idx * 100 // len(loader)))
    # store average loss
    net.update_info('train_loss', running_loss / len(loader))
    return running_loss / len(loader)


# validate
def validate_epoch(net, loader, epoch, criterion,minimize_criterion = True, update_loss = True):
    '''
    function to validate a tuple of networks (and if needed optoimizers) over a given testing loader
    args:
      -net <torch.nn.Module>: neural network to train
      -optim <torch.optim>:  optimizer for neural network
      -loader <torch.dataloader>: data loader object containing LABELLED data
      -epoch <int>: training epoch the nets currently are
      -criterion <torch.nn>: criterion between decoded image and original image
      -minimize_criterion: <bool> if False loss = 1 - criterion(out,input), maximizes criterion (default = True)
      -update_loss <bool>: flag to execute: net.update_info('val_loss', running_loss / len(loader))
    '''

    print("validating MODEL: {}".format(net.get_info('model_name')))
    net.update_info("epoch", epoch)
    net.eval()
    running_loss = 0
    for batch_idx, (img, target) in enumerate(loader):
        img = img.to(device).type(tensortype)
        target = target.to(device).type(tensortype)

        # forward pass
        out = net(target)
        if minimize_criterion:
            loss = criterion(out, img)
        else:
            loss = - criterion(out, img)
        running_loss += loss.item()
        # print some information about where we are in training
        if batch_idx % 10 == 0:
            print("--> {}%".format(batch_idx * 100 // len(loader)))
    # store average loss
    if update_loss:
        net.update_info('val_loss', running_loss / len(loader))
    return running_loss / len(loader)


def save(net,losses, optim):
    '''
    function to save network to a checkpoint, three copies of the network are being saved:
    last.tar, best_training.tar, best_validation.tar

    :param net: <torch.nn.Module> network
    :param losses: <dict> containing training  losses and validation losses
    :param optim: <torch.optim> optimizer used during training

    EXAMPLE TO CONSTRUCT INPUT LOSSES DICTIONARY:

    losses = {}
    losses['train'] = []
    losses['validate'] = []

    or

    losses = {}
    losses['train'] = []
    losses['validate'] = {'A': [], 'B': []}

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

def save_and_display_minibatch(net, ds, idxs, epoch=None, plot = False, save = False):
    '''
    function to save and/or display results of multiple network on a random image of the loader
    :param net: <torch.nn.Module> neural network being trained
    :param ds: <torch.utils.dataset> dataset containing images
    :param epoch (optional): <int> current training epoch
    :param plot: (optional): <bool> flag indicating if plotting the result or only saving it
    :param save: (optional): <bool> flag indicating if saving the image
    '''

    names = ()
    name = net.get_info('model_name')
    names += (name,)
    if epoch is not None:
        if epoch < 10:
            savepath = net.get_info('results_root') + net.get_info('model_configuration')['mode'] + "/" + \
                   ds.get_mode() + "/EPOCH_00{}.png".format(epoch)
        elif epoch >= 10 and epoch < 100:
            savepath = net.get_info('results_root') + net.get_info('model_configuration')['mode'] + "/" + \
                   ds.get_mode() + "/EPOCH_0{}.png".format(epoch)
        else:
            savepath = net.get_info('results_root') + net.get_info('model_configuration')['mode'] + "/" + \
                   ds.get_mode() + "/EPOCH_00{}.png".format(epoch)
    else:
        savepath = net.get_info('results_root') + net.get_info('model_configuration')['mode'] + "/" + \
        ds.get_mode() + "/TEST_IMAGE.png".format(epoch)
    imgs,outs = [],[]
    for idx in idxs:
        img, target = ds[idx]
        img = img.detach().cpu().numpy().squeeze()
        target = torch.tensor(target).unsqueeze(0).to(device).type(tensortype)
        try:
            out = net(target)
        except:
            noise = torch.randn(1, 100, device=device).type(tensortype)
            out = net(noise,target)
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


#function to adjust learning rate:
def adjust_learning_rate(optimizer,lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 25 epochs"""
    lr = lr * (0.1 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_session(net, train_ds,test_ds, model_root, results_root, checkpoint=None, test_idxs=None, train_idxs=None, criterion = None, minimize_criterion = True,
                  START_EPOCH = 0, MAX_EPOCH = 500, lr = None, adjust_lr = False,batch_size = (64,16), num_workers = (8,2),
                  shuffle = (True,False)):
    '''
    function to fully train, validate and save a network (along as some epoch results) given datasets and desired root directories

    :param net: <torch.nn.Module> network to train
    :param train_ds: <torch.utils.data.Dataset> training dataset (images + trackers)
    :param test_ds:  <torch.utils.data.Dataset or tuple of torch.utils.data.Dataset> test dataset(s) (images + trackers)
    :param model_root: <str> path to where to save the model
    :param results_root: <str> path to directory to save epochs' results
    :param checkpoint: <dict> torch checkpoint from which to resume training
    :param test_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of test_ds that we want to plot (with 'save_minibatch()') default: None
    :param train_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of train_ds that we want to plot (with 'save_minibatch()') default: None
    :param criterion: <torch.nn.loss> weight on criterion2 (default = None, default set later)
    :param minimize_criterion: <bool> if False loss = 1 - criterion(out,input), maximizes criterion (default = True)
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

    # determine indices of images to show at each epoch if not given
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
        assert len(test_ds) == 2, "ERROR in train_session: currently developed for max 2 test datasets."
        test_loader = torch.utils.data.DataLoader(test_ds[0], batch_size=batch_size[1], num_workers=num_workers[1], shuffle = shuffle[1])
        test_loader1 = torch.utils.data.DataLoader(test_ds[1], batch_size=batch_size[1], num_workers=num_workers[1], shuffle = shuffle[1])
    else:
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size[1], num_workers=num_workers[1], shuffle = shuffle[1])

    #instanciate criterion if None
    if criterion is None:
        criterion = torch.nn.MSELoss()

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

        net.to(device)
        # print info before starting the training loop
        net.print_info(verbose=True)

        losses['train'] = []
        if isinstance(test_ds, tuple):
            losses['validate'] = {'A': [], 'B': []}
        else:
            losses['validate'] = []
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
        criterion.cuda


    # loop through epochs
    for idx, epoch in enumerate(range(START_EPOCH + 1, MAX_EPOCH + 1)):
        print("\n\n**EPOCH: [{}/{}]**".format(epoch, MAX_EPOCH))
        train_loss = train_epoch(net, optim, train_loader, epoch, criterion, minimize_criterion)
        validate_loss = validate_epoch(net, test_loader, epoch, criterion, minimize_criterion)
        if isinstance(test_ds, tuple):
            validate_loss1 = validate_epoch(net, test_loader1, epoch, criterion, minimize_criterion, update_loss=False)

        # store losses
        losses['train'].append(train_loss)

        if isinstance(test_ds, tuple):
            losses['validate']['A'].append(validate_loss)
            losses['validate']['B'].append(validate_loss1)
        else:
            losses['validate'].append(validate_loss)

        save(net, losses, optim)

        print("\n**EPOCH RESULTS**")
        net.training_summary()
        if adjust_lr and lr > 10e-6:
            lr = lr * (0.1 ** (epoch // 2))
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            print("** current learning rate: {} **".format(lr))
        if isinstance(test_ds, tuple):
            save_and_display_minibatch(net, test_ds[0], test_idxs, epoch, save=True)
        else:
            save_and_display_minibatch(net, test_ds, test_idxs, epoch, save=True)
        save_and_display_minibatch(net, train_ds, train_idxs, epoch, save=True)
    plt.show()


def GAN_training(netG, train_ds,test_ds, model_root, results_root, test_idxs, train_idxs, gen_skip_update_epochs = 1, criterion = None,
                  START_EPOCH = 0, MAX_EPOCH = 500, lr = None, beta1 = 0.5, adjust_lr = False,batch_size = (64,16), num_workers = (8,2),
                  shuffle = (True,False)):
    '''
    function to finetune, validate and save a previously trained generator network via GAN approach.
    given datasets and desired root directories

    :param netG: <torch.nn.Module> Generator network to finetune
    :param train_ds: <torch.utils.data.Dataset> training dataset (images + trackers)
    :param test_ds:  <torch.utils.data.Dataset> test dataset (images + trackers)
    :param model_root: <str> path to where to save the model
    :param results_root: <str> path to directory to save epochs' results
    :param test_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of test_ds that we want to plot (with 'save_minibatch()')
    :param train_idxs: <np.ndarray, shape = (l,)> array containing fixed indices for samples of train_ds that we want to plot (with 'save_minibatch()')
    :param gen_skip_update_epochs: int , number of epoch to skip each time to update the generator
    :param criterion: <torch.nn.loss> weight on criterion2 (default = None, default set later)
    :param START_EPOCH: <int> starting training epoch (default = 0)
    :param MAX_EPOCH: <int> max training epoch (default = 500)
    :param lr: <float> learning rate (default = None, so that optimizer uses its default value)
    :param beta1: <float> beta1 parameter for optimizer (default = 0.5)
    :param adjust_lr: <bool> flag indicating if to use function adjust_learning_rate() (above, default = False)
    :param batch_size: <tuple<int>> tuple with batch sizes for train and test loaders (default = (64,16))
    :param num_workers: <tuple<int>> tuple with number of workers to load train and test datasets (default = (8, 2))
    :param shuffle: <tuple<bool>> tuple indicating whether to shuffle or not train and test loaders (default = (True, False))

    '''

    #check if model_root and results_root directories exist
    assert os.path.isdir(model_root), "ERROR: directory to save the model <model_root: {}> not found.".format(model_root)
    assert os.path.isdir(results_root), "ERROR: directory to save results <results_root: {}> not found.".format(results_root)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size[0], num_workers=num_workers[0], shuffle = shuffle[0])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size[1], num_workers=num_workers[1], shuffle = shuffle[1])

    from architectures.Discriminator import Conditional_Discriminator
    netD = Conditional_Discriminator(z_size=7,latent_image_size=(32,8,8))

    # custom weights initialization called on netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)

    netD.apply(weights_init)
    netG.apply(weights_init)

    #instanciate optimizers
    if lr is not None:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    else:
        optimizerD = torch.optim.Adam(netD.parameters(), betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), betas=(0.5, 0.999))

    #instanciate criterion if None
    if criterion is None:
        criterion = torch.nn.BCELoss()

    #move to GPU
    if use_cuda:
        netG.to(device)
        netD.to(device)
        train_loader.pin_memory = True
        test_loader.pin_memory = True
        cudnn.benchmark = True
        criterion.cuda

    # # save the current info as 'old_info' (so that the new model has all the info of the old model
    # in case we forget where it comes from)

    # info = {}
    # for key in netG.get_info():
    #     info[key] = netG.get_info(key)
    # netG.update_info('old_info', info)

    netG.update_info('model_root', model_root)
    netG.update_info("results_root", results_root)
    netG.update_info("epoch", START_EPOCH)
    netG.update_info("best_val_loss", sys.float_info.max)
    netG.update_info("best_train_loss", sys.float_info.max)
    netG.update_info("val_loss", sys.float_info.max)
    netG.update_info("train_loss", sys.float_info.max)
    #print info before starting the training loop
    netG.print_info(verbose=True)

    # declare losses dict
    losses = {}
    losses['train'] = np.zeros((1, MAX_EPOCH - START_EPOCH))
    losses['validate'] = np.zeros((1, MAX_EPOCH - START_EPOCH))

    # Training Loop

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []

    print("Starting Training Loop...")
    # For each epoch
    for idx, epoch in enumerate(range(START_EPOCH + 1, MAX_EPOCH + 1)):

        #TRAINING
        netD.train()
        netG.train()
        for i, data in enumerate(train_loader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netG.update_info("epoch", epoch)
            netD.zero_grad()
            # Format batch
            b_size = batch_size[0] if i<len(train_ds)//batch_size[0] else len(train_ds)%batch_size[0]
            label = torch.full((b_size,1), real_label, device=device)
            # Forward pass real batch through D
            images,trackers = data[0].to(device).type(tensortype),data[1].to(device).type(tensortype)
            output = netD(images,trackers)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch

            #Generate noise
            noise = torch.randn(b_size, 100, device=device).type(tensortype)
            # Generate fake image batch with G (from tracker info)
            fake = netG(noise, data[1].to(device).type(tensortype))
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach(),data[1].to(device).type(tensortype))
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake,data[1].to(device).type(tensortype))
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            #only optimize the generator evey 'gen_skip_update_epochs' epochs
            if epoch%gen_skip_update_epochs == 0:
                #accumulate gradients
                errG.backward()
                # Update G
                optimizerG.step()
            #get mean score of discriminator on the fake batch
            D_G_z2 = output.mean().item()



            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, MAX_EPOCH, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            netG.update_info('train_loss', errG.item())
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        #generate images
        save_and_display_minibatch(netG, train_ds, train_idxs, epoch, save=True)

        #save netG
        savepath = netG.get_info('model_root') + netG.get_info('model_configuration')['mode'] + "/last.tar"
        torch.save({'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netG.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'loss_G': G_losses,
                    'loss_D': D_losses,
                    'model_info': netG.get_info(),
                    }, savepath)


