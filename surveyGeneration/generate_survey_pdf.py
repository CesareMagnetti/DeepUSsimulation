# generate a survey pdf file with images to evaluate the method

from reportlab.pdfgen import canvas
from PIL import Image

output_folder = '/Users/cesaremagnetti/Documents/BEng_project/surveys/phantom/survey_for_image_quality/'

#       for phantom       #
# image_folder = '/home/ag09/data/iFIND/simulation/results/phantom_images/phantom_survey_images/phantom_survey_images/'
# pdf_output_filename = 'phantom_survey.pdf'
# n_alternatives=3

#       for patient       #
image_folder = '/Users/cesaremagnetti/Documents/BEng_project/surveys/phantom/survey_for_image_quality/survey_images/'
pdf_output_filename = 'patient_survey.pdf'
n_alternatives = 2

# read one image to see the image size
im = Image.open(image_folder + '1.png')

pagesize = (im.size[0] * 1.1, im.size[1] * 1.2)


def addImageToSurvey(canvas, image_folder, image_number, n_alternatives=3):
    canvas.drawImage(image_folder + str(image_number) + '.png', pagesize[0] * 0.05, 0, width=pagesize[0] * 0.9)

    interval = pagesize[0] * 0.9 / (n_alternatives + 1)

    for i in range(n_alternatives):
        form.radio(name='radio' + str(image_number),
                   value='value' + str(i), selected=False,
                   x=pagesize[0] * 0.05 + interval * (i + 1.5), y=pagesize[1] * 0.2,
                   buttonStyle='check',
                   borderStyle='solid', shape='circle',
                   forceBorder=True, fieldFlags='radio')
    canvas.showPage()


def addInstructionsPage(canvas):
    textobject = canvas.beginText()
    textobject.setTextOrigin(pagesize[0] * 0.05, pagesize[1] * 0.9)
    textobject.setFont("Helvetica-Oblique", 14)
    # textobject.textLines('''
    #     Instructions:
    #
    #     Each page shows four images: the first (left) is a real ultraosund image acquired from a phantom.
    #     The other three images are images simulated using different models.
    #
    #     Please choose which one of the simulated images is best, considering not only image quality but
    #     also the features and structures shown.
    #
    #     IMPORTANT: please not that some simulated images look good but they are different from the original
    #     images (they are similar to another image seen during training), and other images that have lower image
    #     quality actually are a better match to the original image. For example, in the case below, image 1 has
    #     the best image quality but in terms of features image 3 is best, so image 3 should be chosen. Image 1
    #     probably is replicating another image seen in training.
    #     ''')

    textobject.textLines('''
        Instructions:

        Each page shows three images: the first (left) is a real ultraosund image acquired from a phantom.
        The other two images are a blurred version of the original image and a blurred simulated image,\n
         the images are not fixed in their column i.e. they are randomly shuffled each time.

        Please choose which one of the two images you think is the blurred original image.
        
        IMPORTANT: please not that some simulated images look good but they are different from the original
        images (they are similar to another image seen during training), and other images that have lower image
        quality actually are a better match to the original image. For example, in the case below, image 1 has
        the best image quality but in terms of features image 3 is best, so image 3 should be chosen. Image 1
        probably is replicating another image seen in training.
        ''')
    canvas.drawText(textobject)
    canvas.drawImage(output_folder + 'sample.png', pagesize[0] * 0.05, 0.15, width=pagesize[0] * 0.9, height=pagesize[1] * 0.3)
    canvas.showPage()


c = canvas.Canvas(output_folder + pdf_output_filename, pagesize=pagesize, bottomup=1)

form = c.acroForm

addInstructionsPage(c)
for i in range(100):
    addImageToSurvey(c, image_folder, i + 1, n_alternatives=n_alternatives)

c.save()
