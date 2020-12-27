'''
create a txt file of N random paired files from a directory, example of terminal run:

python write_random_pair_of_files_to_text.py -r "root-directory" -N 500

will write 500 .mhd and 500 .raw random paired files contained somewhere in <root-directory> to your home directory in the file 'random_names.txt'

Author: Cesare Magnetti
Kings College London, 2019

TO THEN MOVE FILES TO ANOTHER FOLDER USE THE FOLLOWING BASH CODE:

while read filename; do mv ${filename} target/; done < files.txt

where target indicates the directory where you want to move the files
'''

#directory which contains files
import os
from random import shuffle
import argparse

parser = argparse.ArgumentParser(description='Webcam Validation Program')
parser.add_argument('--root','-r',default=None, help='directory containing paired files')
parser.add_argument('--number', '-N', default=None, type= int, help='number of files to output' )
parser.add_argument('--outname', '-n', default = '/home/cm19/random_names.txt', help = 'output file  name')
args = parser.parse_args()
#ARGUMENTS HANDLING
assert args.root is not None and args.number is not None, "--root and --number inputs must be parsed!"

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

if __name__ == '__main__':
    filenames = [os.path.realpath(y) for y in gglob(args.root, '*.*')]
    #get filenames without extension
    names = [f.split('.')[0].strip() for f  in filenames]
    names = list(set(names))
    #randomize the list
    shuffle(names)
    names = names[:args.number]
    one = [f + ".mhd" for f in names]
    two = [f + ".raw" for f in names]
    out = one + two
    out.sort()
    with open(args.outname, 'w') as f:
        for item in out:
            f.write("%s\n" % item)