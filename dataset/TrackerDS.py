'''

custom dataset to load .mhd images and return tracker information as labels
cesare magnetti 2019, King's College London
Requirments:
PyTorch: torch.__version__  = 1.1.0
SimpleITK: sitk.Version() = SimpleITK Version: 1.2.2 (ITK 4.13)
'''

#imports
import SimpleITK as sitk
import numpy as np
import os
from torch.utils.data import Dataset
import math



def quaternion_from_matrix(matrix, isprecise=False):

    """
    Return quaternion from rotation matrix.

    input: matrix --> ROTATION MATRIX TO BE CONVERTED IN QUATERNION FORM
           isprecise --> FLAG, IF TRUE FASTER ALGORITHM IS USED

    output: q -->   quaternion  associated with input rotation matrix
    """

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


#define file formats that will work
IMG_EXTENSIONS = ['.nii.gz', '.nii', '.mha', '.mhd']

def _is_image_file(filename):
    """
    Is the given extension in the filename supported ?
    """
    # FIXME: Need to add all available SimpleITK types!
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_image(fname):
    """
    Load supported image and return the loaded image.
    """
    return sitk.ReadImage(fname)


def save_image(itk_img, fname):
    """
    Save ITK image with the given filename
    """
    sitk.WriteImage(itk_img, fname)


def load_metadata(itk_img, key):
    """
    Load the metadata of the input itk image associated with key.
    """
    return itk_img.GetMetaData(key) if itk_img.HasMetaDataKey(key) else None


def ExtractTrackerMatrixFromFile(fname):
    with open(fname) as f:
        for line in f:
            if "TrackerMatrix =" in line:
                f.close()
                TrackerMatrix = np.array([float(i) for i in line.split(' = ')[1].strip().split(' ')]).reshape(4,4)
                T = TrackerMatrix[:3,-1]
                R = TrackerMatrix[:3,:3]
                Q = quaternion_from_matrix(R)
                # build the target numpy array (t1,t2,t3,q1,q2,q3,q4)
                ret = []
                for t in T: ret.append(t)
                for q in Q: ret.append(q)
                return np.array(ret)
    f.close()
    return None


class TrackerDS(Dataset):
    """
    Arguments
    ---------
    root : string
        Root directory of dataset. The folder should contain all images for each
        mode of the dataset ('train', 'validate', or 'infer'). Each mode-version
        of the dataset should be in a subfolder of the root directory

        The images can be in any ITK readable format (e.g. .mha/.mhd)
        For the 'train' and 'validate' modes, each image should contain a metadata
        key 'Label' in its dictionary/header

    mode : string, (Default: 'train')
        'train', 'validate', or 'infer'
        Loads data from these folders.
        train and validate folders both must contain subfolders images and labels while
        infer folder needs just images subfolder.
    transform : callable, optional
        A function/transform that takes in input itk image or Tensor and returns a
        transformed
        version. E.g, ``transforms.RandomCrop``

    """

    def __init__(self, root, mode='train', transform=None, target_transform=None, classes=None):
        # training set or test set

        assert (mode in ['train', 'validate', 'infer'])

        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(root, 'train')
        elif mode == 'validate':
            self.root = os.path.join(root, 'validate')
        else:
            self.root = os.path.join(root, 'infer') if os.path.exists(os.path.join(root, 'infer')) else root

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

        # Get filenames of all the available images
        self.filenames = [os.path.realpath(y) for y in gglob(self.root, '*.*') if _is_image_file(y)]

        if len(self.filenames) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"
                                                                                  "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.filenames.sort()

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Arguments
        ---------
        index : int
            index position to return the data
        Returns
        -------
        tuple: (image, Tracker) where tracker is a tuple of translation and quaternion rotation parameters
        """
        image = load_image(self.filenames[index])
        Tracker = ExtractTrackerMatrixFromFile(self.filenames[index])

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            Tracker = self.target_transform(Tracker)

        if (self.mode == 'infer') or (Tracker is None):
            return image
        else:
            return image, Tracker

    def __len__(self):
        return len(self.filenames)

    def get_filenames(self):
        return self.filenames

    def get_root(self):
        return self.root




