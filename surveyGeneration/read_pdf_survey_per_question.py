import PyPDF2
import numpy as np
import os
from reportlab.pdfgen import canvas
from PIL import Image

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
output_roots = ['/Users/cesaremagnetti/Documents/BEng_project/survey_responses/phantom/best.txt',
                '/Users/cesaremagnetti/Documents/BEng_project/survey_responses/patient/best.txt']
n_alternatives = 3
decoder,autoencoder,pretrained = [],[],[]
for root, GT_root, output_root in zip(roots,GT_roots,output_roots):
    filenames = [os.path.realpath(y) for y in gglob(root, '*.*') if ".pdf" in y]
    files = [PyPDF2.PdfFileReader(file) for file in filenames]
    fields = [PyPDF2.PdfFileReader(file).getFields() for file in filenames]
    GT = np.loadtxt(GT_root)[:,1:]

    #ASSERT ALL PDFs HAVE THE SAME NUMBER OF PAGES
    assert len(np.unique([f.numPages for f in files])) == 1, "ERROR: PDFs must have the same number of pages"
    answers = []
    out_file = open(output_root, "w")

    #loop through pages
    for idx in range(files[0].numPages-1):
      # instanciate choices
      choices = np.zeros((1, n_alternatives))
      for field in fields:
          #get the current page information of each document
          f = field['radio{}'.format(idx+1)]
          if "/V" in f:
              value = f["/V"]
              # check if field wasn't left empty and get the survey choice
              if '/value' in value:
                  if GT[idx,int(value[-1])] == 1:
                      choices[0, 0] += 1
                  elif GT[idx,int(value[-1])] == 2:
                      choices[0, 1] += 1
                  elif GT[idx,int(value[-1])] == 3:
                      choices[0, 2] += 1
              else:
                  continue
      #append the architecture that performed better
      out_file.write("{}\n".format(np.argmax(choices)+1))
      answers.append(np.argmax(choices)+1)

    #to count responses run the following code
    decoder_count,autoencoder_count,pretrained_count = 0,0,0
    for a in answers:
        if a==1:
            decoder_count+=1
        elif a==2:
            autoencoder_count+=1
        elif a==3:
            pretrained_count+=1
    decoder.append(decoder_count/len(answers)*100)
    autoencoder.append(autoencoder_count/len(answers)*100)
    pretrained.append(pretrained_count/len(answers)*100)
    print("decoder architecture perfprmed best {:.2f}% of times,\nautoencoder architecture perfprmed best {:.2f}% of times,\n"
          "pretrained decoder architecture perfprmed best {:.2f}% of times".format(decoder_count/len(answers)*100,
                                                                                   autoencoder_count/len(answers)*100,
                                                                                   pretrained_count/len(answers)*100))

print(decoder,autoencoder,pretrained)