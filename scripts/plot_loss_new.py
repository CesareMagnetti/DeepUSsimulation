import torch
from matplotlib import pyplot as plt
import os

def gglob(path, regexp=None):
    """Recursive glob
    """
    import fnmatch
    import os
    matches = []
    if regexp is None:
        regexp = '*'
    for root, dirnames, filenames in os.walk(path, followlinks=True):
        for filename in fnmatch.filter(filenames, regexp):
            matches.append(os.path.join(root, filename))
    return matches

#DEFINE ROOT DIRECTORY, MODEL NAME AND ARCHITECTURE TYPE#########################

roots = [
        "/Users/cesaremagnetti/linode/models/DECODER/final_models/phantom/DECODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/resize_conv/",
        "/Users/cesaremagnetti/linode/models/DECODER/final_models/phantom/DECODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/pretrained/resize_conv/",
        #"/Users/cesaremagnetti/linode/models/DECODER/final_models/phantom/other_sizes/44/DECODER__LINEAR_7_32_256_512_1024_2048__CONV_128_32_16_8_4_1/resize_conv/",
        "/Users/cesaremagnetti/linode/models/AUTOENCODER/final_models/phantom/AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/resize_conv/",
        "/Users/cesaremagnetti/different_variationals/VARIATIONAL_AUTOENCODER/final_models/phantom/"
        "VARIATIONAL_AUTOENCODER__LINEAR_7_32_256_512_1024_2048__CONV_32x5_1/BETA_1e-03/resize_conv/",
        ]
NAME = "last"

##################################################################################

train_losses,val_losses,epochs,names = [],[],[],[]
flag = False
# Get filenames of all the available models of that mode

for root in roots:
    filenames = [os.path.realpath(y) for y in gglob(root, '*.*') if NAME in y]
    for file in filenames:
        if NAME not in file:
            continue
        checkpoint = torch.load(file, map_location='cpu')
        epochs.append(checkpoint['model_info']['epoch'])
        names.append(checkpoint['model_info']['model_name'])
        train_losses.append(checkpoint['train_loss_hist'])
        val_losses.append(checkpoint['validation_loss_hist'])
        if 'A' in checkpoint['validation_loss_hist']:
            flag = True
        # layers.append(len(checkpoint['model_info']['model_configuration']['nlinear'])\
        #          + len(checkpoint['model_info']['model_configuration']['stride']))

names = ["decoder", "pretrained decoder", "autoencoder: K = 1", "variational autoencoder: " + r"$\beta \ = 0.001, K \ =1"]
print(val_losses)
fig = plt.figure()
if flag:
    fig.suptitle('LOSSES FOR CHOSEN {}\n(solid line -> training loss,   dashed line -> validation loss: region A,'
                 ' dotted line -> validation loss: region B)'.format("ARCHITECTURE" if len(epochs) == 1 else "ARCHITECTURES"),
                 fontsize=10)
else:
    fig.suptitle('LOSSES FOR CHOSEN {}\n(solid line -> training loss,   dashed line -> validation loss)'.format(
                 "ARCHITECTURE" if len(epochs) == 1 else "ARCHITECTURES"), fontsize=10)
ax = plt.gca()

for train_loss, val_loss, epoch, name, root in zip(train_losses,val_losses,epochs, names, roots):
    if 'DECODER' in name or 'refined_freezing_encoder' in root:
        # PLOT LOSSES FOR DECODER
        if isinstance(val_loss, dict):
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch + 1 - len(train_loss), epoch + 1), train_loss, c=color, label=name)
            plt.plot(range(epoch + 1 - len(val_loss['A']), epoch + 1), val_loss['A'], c=color, ls='--')
            plt.plot(range(epoch + 1 - len(val_loss['B']), epoch + 1), val_loss['B'], c=color, ls=':')
            plt.legend()
        else:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch+1-len(train_loss), epoch+1),train_loss, c = color, label = name)
            plt.plot(range(epoch+1-len(val_loss), epoch+1), val_loss, c = color, ls = '--')
            plt.legend()

    elif 'AUTOENCODER' in name and not 'refined_freezing_encoder' in root and not 'VARIATIONAL_AUTOENCODER' in root:
        #PLOT LOSSES FOR AUTOENCODER
        if 'A' in val_loss:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch + 1 - len(train_loss['autoencoder']), epoch + 1), train_loss['autoencoder'],
                     c=color, label=name + "__AUTOENCODED_IMAGE")
            plt.plot(range(epoch + 1 - len(val_loss['A']['autoencoder']), epoch + 1), val_loss['A']['autoencoder'],
                     c=color, ls='--')
            plt.plot(range(epoch + 1 - len(val_loss['B']['autoencoder']), epoch + 1), val_loss['B']['autoencoder'],
                     c=color, ls=':')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch + 1 - len(train_loss['decoder']), epoch + 1), train_loss['decoder'],
                     c=color, label=name + "__DECODED_IMAGE")
            plt.plot(range(epoch + 1 - len(val_loss['A']['decoder']), epoch + 1), val_loss['A']['decoder'], c=color,
                     ls='--')
            plt.plot(range(epoch + 1 - len(val_loss['B']['decoder']), epoch + 1), val_loss['B']['decoder'], c=color,
                     ls=':')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch + 1 - len(train_loss['encoder']), epoch + 1), train_loss['encoder'], c=color,
                     label=name + "__TRACKER")
            plt.plot(range(epoch + 1 - len(val_loss['A']['encoder']), epoch + 1), val_loss['A']['encoder'], c=color, ls='--')
            plt.plot(range(epoch + 1 - len(val_loss['B']['encoder']), epoch + 1), val_loss['B']['encoder'], c=color, ls=':')
            #color = next(ax._get_lines.prop_cycler)['color']
            # plt.plot(range(epoch+1-len(train_loss['autoencoder']), epoch+1), train_loss['encoder'] + train_loss['autoencoder'],
            #          c = color, label=name + "__TOTAL")
            # plt.plot(range(epoch+1-len(val_loss['A']['autoencoder']), epoch+1), val_loss['encoder'] + val_loss['A']['autoencoder'],
            #          c = color, ls = '--')
            # plt.plot(range(epoch+1-len(val_loss['B']['autoencoder']), epoch+1), val_loss['encoder'] + val_loss['B']['autoencoder'],
            #          c = color, ls = ':')
            plt.legend()
        else:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch+1-len(train_loss['autoencoder']), epoch+1), train_loss['autoencoder'],
                     c = color, label = name + "__AUTOENCODED_IMAGE")
            plt.plot(range(epoch+1-len(val_loss['autoencoder']), epoch+1), val_loss['autoencoder'], c = color,
                     ls ='--')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch+1-len(train_loss['decoder']), epoch+1), train_loss['decoder'], c = color,
                     label=name + "__DECODED_IMAGE")
            plt.plot(range(epoch+1-len(val_loss['decoder']), epoch+1), val_loss['decoder'], c= color, ls ='--')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch+1-len(train_loss['encoder']), epoch+1), train_loss['encoder'], c = color, label=name+"__TRACKER")
            plt.plot(range(epoch+1-len(val_loss['encoder']), epoch+1), val_loss['encoder'],c = color, ls ='--')
            # color = next(ax._get_lines.prop_cycler)['color']
            # plt.plot(range(epoch+1-len(train_loss['autoencoder']), epoch+1), train_loss['encoder'] + train_loss['autoencoder'],
            #          c = color, label=name + "__TOTAL")
            # plt.plot(range(epoch+1-len(val_loss['autoencoder']), epoch+1), val_loss['encoder'] + val_loss['autoencoder'],
            #          c = color, ls = '--')
            plt.legend()
    elif 'VARIATIONAL_AUTOENCODER' in name and not 'refined_freezing_encoder' in root:
        #PLOT LOSSES FOR AUTOENCODER
        if 'A' in val_loss:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch + 1 - len(train_loss['autoencoder']), epoch + 1), train_loss['autoencoder'],
                     c=color, label=name + "__AUTOENCODED_IMAGE")
            plt.plot(range(epoch + 1 - len(val_loss['A']['autoencoder']), epoch + 1), val_loss['A']['autoencoder'],
                     c=color, ls='--')
            plt.plot(range(epoch + 1 - len(val_loss['B']['autoencoder']), epoch + 1), val_loss['B']['autoencoder'],
                     c=color, ls=':')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch + 1 - len(train_loss['decoder']), epoch + 1), train_loss['decoder'],
                     c=color, label=name + "__DECODED_IMAGE")
            plt.plot(range(epoch + 1 - len(val_loss['A']['decoder']), epoch + 1), val_loss['A']['decoder'], c=color,
                     ls='--')
            plt.plot(range(epoch + 1 - len(val_loss['B']['decoder']), epoch + 1), val_loss['B']['decoder'], c=color,
                     ls=':')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch + 1 - len(train_loss['encoder']), epoch + 1), train_loss['encoder'], c=color,
                     label=name + "__TRACKER")
            plt.plot(range(epoch + 1 - len(val_loss['A']['encoder']), epoch + 1), val_loss['A']['encoder'], c=color, ls='--')
            plt.plot(range(epoch + 1 - len(val_loss['B']['encoder']), epoch + 1), val_loss['B']['encoder'], c=color, ls=':')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch + 1 - len(train_loss['KLD']), epoch + 1), train_loss['KLD'], c=color,
                     label=name + "__KLD")
            plt.plot(range(epoch + 1 - len(val_loss['A']['KLD']), epoch + 1), val_loss['A']['KLD'], c=color, ls='--')
            plt.plot(range(epoch + 1 - len(val_loss['B']['KLD']), epoch + 1), val_loss['B']['KLD'], c=color, ls=':')
            #color = next(ax._get_lines.prop_cycler)['color']
            # plt.plot(range(epoch+1-len(train_loss['autoencoder']), epoch+1), train_loss['encoder'] + train_loss['autoencoder'],
            #          c = color, label=name + "__TOTAL")
            # plt.plot(range(epoch+1-len(val_loss['A']['autoencoder']), epoch+1), val_loss['encoder'] + val_loss['A']['autoencoder'],
            #          c = color, ls = '--')
            # plt.plot(range(epoch+1-len(val_loss['B']['autoencoder']), epoch+1), val_loss['encoder'] + val_loss['B']['autoencoder'],
            #          c = color, ls = ':')
            plt.legend()
        else:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch+1-len(train_loss['autoencoder']), epoch+1), train_loss['autoencoder'],
                     c = color, label = name + "__AUTOENCODED_IMAGE")
            plt.plot(range(epoch+1-len(val_loss['autoencoder']), epoch+1), val_loss['autoencoder'], c = color,
                     ls ='--')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch+1-len(train_loss['decoder']), epoch+1), train_loss['decoder'], c = color,
                     label=name + "__DECODED_IMAGE")
            plt.plot(range(epoch+1-len(val_loss['decoder']), epoch+1), val_loss['decoder'], c= color, ls ='--')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch+1-len(train_loss['encoder']), epoch+1), train_loss['encoder'], c = color, label=name+"__TRACKER")
            plt.plot(range(epoch+1-len(val_loss['encoder']), epoch+1), val_loss['encoder'],c = color, ls ='--')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch+1-len(train_loss['KLD']), epoch+1), train_loss['KLD'], c = color, label=name+"__KLD")
            plt.plot(range(epoch+1-len(val_loss['KLD']), epoch+1), val_loss['KLD'],c = color, ls ='--')
            # color = next(ax._get_lines.prop_cycler)['color']
            # plt.plot(range(epoch+1-len(train_loss['autoencoder']), epoch+1), train_loss['encoder'] + train_loss['autoencoder'],
            #          c = color, label=name + "__TOTAL")
            # plt.plot(range(epoch+1-len(val_loss['autoencoder']), epoch+1), val_loss['encoder'] + val_loss['autoencoder'],
            #          c = color, ls = '--')
            plt.legend()
plt.show()