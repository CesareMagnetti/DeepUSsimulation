import torch
import glob
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

roots = [#"/Users/cesaremagnetti/Documents/BEng_project_repo/models/DECODER/DeeperDec/noDropout/MSELoss/bigger_latent_image/" \
         #"DECODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/deconv",

         "/home/cm19/BEng_project/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/" \
         "AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/pretraining/deconv",]

# roots = [#"/Users/cesaremagnetti/Documents/BEng_project_repo/models/DECODER/DeeperDec/noDropout/MSELoss/bigger_latent_image/" \
#          #"incomplete_dataset/X_point_-30_-0_-350_X_radius_60/DECODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/",
#
#          "/Users/cesaremagnetti/Documents/BEng_project_repo/models/AUTOENCODER/DeeperAuto/noDropout/MSELoss/bigger_latent_image/" \
#          "incomplete_dataset/X_point_-30_-0_-350_X_radius_60/AUTOENCODER__LINEAR_7_32_512_1024_2048__CONV_32x6_1/"]
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



fig = plt.figure()
if flag:
    fig.suptitle('LOSSES FOR CHOSEN {}\n(solid line -> training loss,   dashed line -> validation loss: region A,'
                 ' dotted line -> validation loss: region B)'.format("ARCHITECTURE" if len(epochs) == 1 else "ARCHITECTURES"),
                 fontsize=10)
else:
    fig.suptitle('LOSSES FOR CHOSEN {}\n(solid line -> training loss,   dashed line -> validation loss)'.format(
                 "ARCHITECTURE" if len(epochs) == 1 else "ARCHITECTURES"), fontsize=10)
ax = plt.gca()

for train_loss, val_loss, epoch, name in zip(train_losses,val_losses,epochs, names):
    if 'DECODER' in name:
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
    elif 'AUTOENCODER' in name:
        #PLOT LOSSES FOR AUTOENCODER
        if 'A' in val_loss:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch + 1 - len(train_loss['img']['from_img']), epoch + 1), train_loss['img']['from_img'],
                     c=color, label=name + "__AUTOENCODED_IMAGE")
            plt.plot(range(epoch + 1 - len(val_loss['A']['img']['from_img']), epoch + 1), val_loss['A']['img']['from_img'],
                     c=color, ls='--')
            plt.plot(range(epoch + 1 - len(val_loss['B']['img']['from_img']), epoch + 1), val_loss['B']['img']['from_img'],
                     c=color, ls=':')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch + 1 - len(train_loss['img']['from_z']), epoch + 1), train_loss['img']['from_z'],
                     c=color, label=name + "__DECODED_IMAGE")
            plt.plot(range(epoch + 1 - len(val_loss['A']['img']['from_z']), epoch + 1), val_loss['A']['img']['from_z'], c=color,
                     ls='--')
            plt.plot(range(epoch + 1 - len(val_loss['B']['img']['from_z']), epoch + 1), val_loss['B']['img']['from_z'], c=color,
                     ls=':')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(epoch + 1 - len(train_loss['z']), epoch + 1), train_loss['z'], c=color,
                     label=name + "__TRACKER")
            plt.plot(range(epoch + 1 - len(val_loss['A']['z']), epoch + 1), val_loss['A']['z'], c=color, ls='--')
            plt.plot(range(epoch + 1 - len(val_loss['B']['z']), epoch + 1), val_loss['B']['z'], c=color, ls=':')
            #color = next(ax._get_lines.prop_cycler)['color']
            # plt.plot(range(epoch+1-len(train_loss['img']['from_img']), epoch+1), train_loss['z'] + train_loss['img']['from_img'],
            #          c = color, label=name + "__TOTAL")
            # plt.plot(range(epoch+1-len(val_loss['A']['img']['from_img']), epoch+1), val_loss['z'] + val_loss['A']['img']['from_img'],
            #          c = color, ls = '--')
            # plt.plot(range(epoch+1-len(val_loss['B']['img']['from_img']), epoch+1), val_loss['z'] + val_loss['B']['img']['from_img'],
            #          c = color, ls = ':')
            plt.legend()
        else:
            if isinstance(train_loss, dict):
                color = next(ax._get_lines.prop_cycler)['color']
                plt.plot(range(epoch+1-len(train_loss['img']['from_img']), epoch+1), train_loss['img']['from_img'],
                         c = color, label = name + "__AUTOENCODED_IMAGE")
                plt.plot(range(epoch+1-len(val_loss['img']['from_img']), epoch+1), val_loss['img']['from_img'], c = color,
                         ls ='--')
                color = next(ax._get_lines.prop_cycler)['color']
                plt.plot(range(epoch+1-len(train_loss['img']['from_z']), epoch+1), train_loss['img']['from_z'], c = color,
                         label=name + "__DECODED_IMAGE")
                plt.plot(range(epoch+1-len(val_loss['img']['from_z']), epoch+1), val_loss['img']['from_z'], c= color, ls ='--')
                color = next(ax._get_lines.prop_cycler)['color']
                plt.plot(range(epoch+1-len(train_loss['z']), epoch+1), train_loss['z'], c = color, label=name+"__TRACKER")
                plt.plot(range(epoch+1-len(val_loss['z']), epoch+1), val_loss['z'],c = color, ls ='--')
                # color = next(ax._get_lines.prop_cycler)['color']
                # plt.plot(range(epoch+1-len(train_loss['img']['from_img']), epoch+1), train_loss['z'] + train_loss['img']['from_img'],
                #          c = color, label=name + "__TOTAL")
                # plt.plot(range(epoch+1-len(val_loss['img']['from_img']), epoch+1), val_loss['z'] + val_loss['img']['from_img'],
                #          c = color, ls = '--')
                plt.legend()
            else:
                color = next(ax._get_lines.prop_cycler)['color']
                plt.plot(range(epoch + 1 - len(train_loss), epoch + 1), train_loss,c=color, label=name + "__pre-train loss")
                plt.plot(range(epoch + 1 - len(val_loss), epoch + 1), val_loss,c=color,ls='--', label=name + "__pre-validate loss")
                plt.legend()


#elif architecture == 'AUTOENCODER':
    #PLOT LOSSES FOR AUTOENCODER
    # plt.plot(range(epoch+1-len(train_losses_hist['img']['from_img']), epoch+1), train_losses_hist['img']['from_img'],'k', label = "loss_img_from_img_train")
    # plt.plot(range(epoch+1-len(train_losses_hist['img']['from_img']), epoch+1), val_losses_hist['img']['from_img'], 'k--', label="loss_img_from_img_val")
    # plt.plot(range(epoch+1-len(train_losses_hist['img']['from_img']), epoch+1), train_losses_hist['img']['from_z'], 'b', label="loss_img_from_z_train")
    # plt.plot(range(epoch+1-len(train_losses_hist['img']['from_img']), epoch+1), val_losses_hist['img']['from_z'], 'b--', label="loss_img_from_z_val")
    # plt.plot(range(epoch+1-len(train_losses_hist['img']['from_img']), epoch+1), train_losses_hist['z'], 'r', label="loss_z_train")
    # plt.plot(range(epoch+1-len(train_losses_hist['img']['from_img']), epoch+1), val_losses_hist['z'],'r--', label="loss_z_val")
    # # plt.plot(range(epoch+1-len(train_losses_hist['img']['from_img']), epoch+1), train_losses_hist['z'] + train_losses_hist['img']['from_img'], 'y', label="total_training_loss")
    # # plt.plot(range(epoch+1-len(train_losses_hist['img']['from_img']), epoch+1), val_losses_hist['z'] + val_losses_hist['img']['from_img'], 'y--', label="total_validation_loss")
    # plt.title(name)
    # plt.legend()

# elif architecture == 'VARIATIONAL_AUTOENCODER':
#     #PLOT LOSSES FOR VARIATIONAL AUTOENCODER
#     for idx,(train_loss,val_loss,name) in enumerate(zip(train_losses_hist,val_losses_hist,names)):
#         axs[idx].plot(range(len(train_loss['img']['from_img'])), train_loss['img']['from_img'],'k', label = "loss_img_from_img_train")
#         axs[idx].plot(range(len(val_loss['img']['from_img'])), val_loss['img']['from_img'], 'k--', label="loss_img_from_img_val")
#         axs[idx].plot(range(len(train_loss['img']['from_z'])), train_loss['img']['from_z'], 'b', label="loss_img_from_z_train")
#         axs[idx].plot(range(len(val_loss['img']['from_z'])), val_loss['img']['from_z'], 'b--', label="loss_img_from_z_val")
#         axs[idx].plot(range(len(train_loss['z'])), train_loss['z'], 'r', label="loss_z_train")
#         axs[idx].plot(range(len(val_loss['z'])), val_loss['z'],'r--', label="loss_z_val")
#         axs[idx].plot(range(len(train_loss['KLD'])), train_loss['KLD'], 'g', label="loss_KLD_train")
#         axs[idx].plot(range(len(val_loss['KLD'])), val_loss['KLD'],'g--', label="loss_KLD_val")
#         axs[idx].title.set_text(name)
#         axs[idx].legend()

plt.show()