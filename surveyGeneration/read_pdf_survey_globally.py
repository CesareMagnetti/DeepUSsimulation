import PyPDF2
import numpy as np
import os
from matplotlib import pyplot as plt

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

roots = ['/Users/cesaremagnetti/Documents/BEng_project/survey_responses/phantom/responses/',
         '/Users/cesaremagnetti/Documents/BEng_project/survey_responses/patient/responses/']

GT_roots = ['/Users/cesaremagnetti/Documents/BEng_project/survey_responses/phantom/phantom_GT.txt',
            '/Users/cesaremagnetti/Documents/BEng_project/survey_responses/patient/patient_GT.txt']

decoder, autoencoder, pretrained_decoder = {"phantom": [], "patient": []},{"phantom": [], "patient": []},{"phantom": [], "patient": []}
TOTAL = []
for root, GT_root, key in zip(roots, GT_roots, decoder):

    #get all pdf responses from root
    filenames = [os.path.realpath(y) for y in gglob(root, '*.*') if '.pdf' in y]

    #get GT as numpy array
    f = open(GT_root)
    GT = []
    for line in f:
        temp = line.split(' ')[1:]
        temp = [int(t.strip()) for t in temp]
        GT.append(temp)
    GT = np.array(GT)
    f.close()

    #count total votes
    total = 0
    #get answers from pdf
    for file in filenames:
        f = PyPDF2.PdfFileReader(file)
        ff = f.getFields()

        #answers = np.zeros((len(ff),n_alternatives))
        choices = []
        for i, field_name in enumerate(ff, start=0):
            field_i = ff[field_name]
            if "/V" in field_i:
                value = field_i['/V']
                #check if field wasn't left empty
                if '/value' in value:
                    chosen = int(value[-1])
                    #answers[i, chosen] =1
                    choices.append(chosen)
                #if did not respond use -1 as flag
                else:
                    choices.append(-1)

        # setting up count variables
        count_decoder, count_autoencoder, count_pretrained_decoder = 0, 0, 0
        for gt, choice in  zip(GT, choices):
            #if unanswered question skip to next one
            if choice == -1:
                continue
            total+=1
            temp = gt[choice]
            if temp == 1:
                count_decoder +=1
            elif temp == 2:
                count_autoencoder +=1
            elif temp == 3:
                count_pretrained_decoder +=1

        #append results to their containers
        decoder[key].append(count_decoder)
        autoencoder[key].append(count_autoencoder)
        pretrained_decoder[key].append(count_pretrained_decoder)
    #append total votes
    TOTAL.append(total)
    # decoder.append(count_decoder/total*100)
    # autoencoder.append(count_autoencoder/total*100)
    # pretrained_decoder.append(count_pretrained_decoder/total*100)

avgs, stds = {"phantom": [], "patient": []}, {"phantom": [], "patient": []}

for key1, key2, key3, tot in zip(decoder, autoencoder, pretrained_decoder, TOTAL):
    assert key1 == key2 and key2 == key3 and key1 == key3, "ERROR: keys should be the same"
    #get percentage of votes
    avgs[key1].append(np.sum(decoder[key1])/tot*100)
    avgs[key1].append(np.sum(autoencoder[key1])/tot*100)
    avgs[key1].append(np.sum(pretrained_decoder[key1])/tot*100)
    #get stds
    stds[key1].append(np.std(decoder[key1]))
    stds[key1].append(np.std(autoencoder[key1]))
    stds[key1].append(np.std(pretrained_decoder[key1]))

print(avgs)

objects = ('decoder', 'autoencoder', 'pretrained\ndecoder')
y_pos = np.array([0.1,0.2, 0.3])

fig, axs = plt.subplots(1,2)
plt.sca(axs[0])
plt.yticks(y_pos, objects)
axs[0].barh(y_pos, avgs["phantom"], xerr = stds["phantom"], height = 0.05)
axs[0].set_xlabel('% favorable ratings')
axs[0].title.set_text('Phantom Data')
plt.sca(axs[1])
plt.yticks([], [])
axs[1].barh(y_pos, avgs["patient"],xerr = stds["phantom"], height = 0.05)
axs[1].set_xlabel('% favorable ratings')
axs[1].title.set_text('Real Patient Data')

plt.show()
