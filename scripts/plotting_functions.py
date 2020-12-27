from matplotlib import pyplot as plt
import numpy as np

# plot tensor image
def plotImage(im, tracker):
    '''
    simple function to plot a tensor image along with its tracker information
    Args:
      -im <torch.tensor> shape(1,1,R,C) input image to plot
      -tracker <torch.tensor> shape <1,C> tracker information
    '''

    im = im.cpu().detach().numpy().squeeze()
    tracker = tracker.cpu().detach().numpy().squeeze()
    T, Q = tracker[:3], tracker[3:]

    fig = plt.figure()
    plt.imshow(im, interpolation='nearest')
    plt.title("T: %.4f , %.4f , %.4f\nQ: %.4f , %.4f , %.4f , %.4f" % (T[0], T[1], T[2], Q[0], Q[1], Q[2], Q[3]))
    plt.show()


def plotImageAndResult(im, res):
    '''
    simple function to plot initial image, generated image and their squared difference
    Args:
      -im <torch.tensor> (shape(1,1,R,C)): input image to plot
      -tracker <torch.tensor> (shape (1,C)): tracker information
    '''
    im = im.cpu().numpy().squeeze()
    res = res.cpu().numpy().squeeze()
    er = np.sqrt(np.sum((res - im) ** 2, axis=0) / 3).squeeze()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(np.transpose(im, (1, 2, 0)), interpolation='nearest')
    ax2.imshow(np.transpose(res, (1, 2, 0)), interpolation='nearest')
    ax3.imshow(er, interpolation='nearest', vmin=0, vmax=1)


def plot_results(im, outs, epoch=None, names=None):
    '''
    simple function to show initial image, and a variable amount of generated images for the current training epoch
    Args:
      -im <torch.tensor> (shape(1,1,R,C)): input image
      -outs <list<numpy.array (shape(R,C))>>: list of outputs as numpy 2d arrays (shape(R,C))
      -epoch <int>: training epoch
      -names <tuple<string>>: tuple containing the names of the models which generated the images
    '''
    l = len(outs)
    if names is None:
        names = ("",) * l
    else:
        assert l == len(
            names), "ERROR: given list of model names to load must have the same length as generated outputs!"

    im = im.detach().cpu().numpy().squeeze()
    # create subplot
    print("\n\t\t\t\t\t\t\t*EPOCH {}*\n".format(epoch))
    fig, axes = plt.subplots(1, l + 1, figsize=(20, 80))
    fig.tight_layout()

    axes[0].imshow(im, interpolation='nearest')
    axes[0].axis('off')
    axes[0].title.set_text('image')
    for ax, name, out in zip(axes[1:], names, outs):
        # out = out.detach().cpu().numpy().squeeze()
        ax.imshow(out, interpolation='nearest')
        ax.axis('off')
        ax.title.set_text(name)
    plt.show()
