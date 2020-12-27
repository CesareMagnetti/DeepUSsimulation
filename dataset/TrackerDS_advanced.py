'''
custom dataset to load .mhd images along as their tracker information.
it is possible to specify the length L of the wanted dataset, in which case,
L random samples will be returned. (if L> 'total number of samples in the directory' all available samples will be returned)
you also have the possibility of discarding a spherical region in the data
distribution providing a selection point (x,y,z) and a radius.
all samples coming from that region won't be loaded

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
import random
from torchvision.transforms import Compose

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


def ExtractKeyFromFile(fname, key):
    with open(fname) as f:
        for line in f:
            if key in line:
                f.close()
                label = line.split(' = ')[1].strip()
                return np.array(label)
    f.close()
    return None


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
            # elif "quaternion =" in line:
            #     f.close()
            #     TrackerMatrix = np.array([float(i) for i in line.split(' = ')[1].strip().split(' ')])
    f.close()
    return None


class TrackerDS_advanced(Dataset):
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
        version. E.g, ``tensor_transforms.RandomCrop``
    target_transform : callable, optional
        A function/transform that takes in input np.array or Tensor and returns a
        transformed
        version. E.g, ``tensor_transforms.CostantRescale``
    N_samples : int, optional
        if integer N_samples given, then N_samples from the dataset will be randomly selected.
        i.e. it will create a dataset random subset of size N_samples
    random_seed : int, optional
        random seed for random sampling when creating a random subset of the data of size N_samples.
        i.e. the same random seed will return the same subset of data
    X_point : tuple, optional
        centre of the spherical sub-region of the dataset to discard or keep, depending on value of 'reject_inside_radius'
    X_radius : int, optional
        the radius of the spherical sub-region of the dataset to discard or keep, depending on value of 'reject_inside_radius'
    reject_inside_radius : bool, optional (default: False)
        if True Points inside the spherical region are discarded, if False points outside the spherical regiorn are discarded
    additional_labels : bool, optional (default: False)
        if True the following additional labels are added to the target vector:
            - FocusDepth
            - SectorAngle
            - SectorWidth
            - SectorStartDepth
            - SectorStopDepth
            - FocalStartDepth
            - FocalStopDepth
            - DepthOfScanField
            - DNLLayerTimeTag/DNLTimestamp/LayerTimeTag/LocalTimestamp/TrackerTimestamp      not working: TransducerTimestamp

    """

    def __init__(self, root, mode='train', transform=None, target_transform=None, N_samples = None, random_seed  =None,
                 X_point = None, X_radius = None, reject_inside_radius = True, additional_labels  = None, filter_class = None):
        '''
        initialisation function for the dataset

        :param root : os.path, root directory containing data, must contain these subfulders: (train/ validate/ infer/)
        :param mode : string, either "train"/"validate"/"infer", sets which set to load (default: "train")
        :param transform : callable, transform on data (default: None)
                         Example:
                         transform = torchtransforms.Compose([resample,
                                     tonumpy,
                                     totensor,
                                     crop,
                                     resize,
                                     rescale])
        :param target_transform: callable, transform on target (default: None)
        :param N_samples: int, number of samples to sample (default: None)
        :param random_seed: int, random seed for repeatability of random sampling of N_sample (default: None)
        :param X_point: tuple<float or int>, XYZ coordinates of the central point of the region to be discarded (default: None)
                      Example:
                      X_point=(-30,-0,-350), X_radius = 30
        :param X_radius: int or float, radius of the spherical region to be discarded (default: None)
        :param reject_inside_radius: bool, if true points inside spherical region are discarded, if false points outside
                                     outside spherical region are discarded. (default: True)
        :param additional_labels: tuple<string>, if true additional labels are retrieved from the .mhd files
                                Example:
                                additional_labels = ('FocusDepth','SectorAngle','SectorWidth','SectorStartDepth',
                                                     'SectorStopDepth', 'FocalStartDepth','FocalStopDepth',
                                                     'DepthOfScanField','TrackerTimestamp')
        :param filter_class: string or tuple<string>, if given, it will filter only images belonging of that class(es) (i.e. "Thorax")
                             Note that the value of 'filter_class' MUST be contained in the image path for a sample
                             to be selected. Make sure that images are stored in the correct folders!
                             Example:
                             path: /home/cm19/BEng_project/data/patient_data/iFIND00366_for_simulator/train/Thorax/file.mhd
                             filter_class = "Thorax" or ("Thorax","Head",..,"Abdomen")
        '''

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

        # Get filenames of all the available images and sort them
        self.filenames = [os.path.realpath(y) for y in gglob(self.root, '*.*') if _is_image_file(y)]
        #sort files, they will be shuffled by the data loader
        self.filenames.sort()

        #discard files not belonging to the requested class
        if filter_class is not None:
            if isinstance(filter_class,tuple):
                print("filtering requested classes...")
                #only keep files not belonging to the requested classes
                self.filenames = [y for y in self.filenames if any(c in y for c in filter_class)]
            else:
                print("filtering {} samples".format(filter_class))
                #only keep files not belonging to the requested class
                self.filenames = [y for y in self.filenames if filter_class in y]

        # check that filenames is not empty
        if len(self.filenames) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"
                                                                                  "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        # sample N_samples samples if needed
        if N_samples is not None and N_samples < len(self.filenames):
            print("sampling {} random files from the dataset...".format(N_samples))
            random.seed(random_seed)
            self.filenames = random.sample(self.filenames, N_samples)

        # create a hole in the dataset if needed
        if X_point is not None and X_radius is not None:
            assert (isinstance(X_point, tuple) and len(X_point) == 3), \
                "ERROR in TrackerDS_advanced: X_point parameter should be a tuple of length 3 (x,y,z)"
            if reject_inside_radius:
                print("creating a spherical hole in the dataset with centre ({},{},{}) and radius {}".format(*X_point, X_radius))
            else:
                print("creating a spherical dataset with centre ({},{},{}) and radius {}".format(*X_point, X_radius))

            fnames = []
            for fname in self.filenames:
                Tracker = ExtractTrackerMatrixFromFile(fname)
                distance = np.sqrt(np.sum((Tracker[:3]-X_point)**2))
                #reject points inside that radius or outside (depending on parameter: reject_inside_radius)
                if reject_inside_radius:
                    if ( distance < X_radius):
                       continue
                else:
                    if ( distance > X_radius):
                       continue

                fnames.append(fname)
            self.filenames = fnames

        #store data tranform
        self.transform = transform
        #store target transform
        self.target_transform = target_transform
        #original labels retrieved
        self.labels = ('X','Y','Z','Q1','Q2','Q3','Q4',)
        #original + additional labels retrieved
        if additional_labels is not None:
            assert isinstance(additional_labels, tuple), "ERROR: 'additional_labels' parameter must be a tuple of strings."
            print(str("retriecing the following additional labels for each samples: " +
                      "{}, "*(len(additional_labels)-1) + "{}").format(*additional_labels))
            #add the following labels
            self.labels+=additional_labels

        #store only additional labels for easier usage in self.get_item()
        self.additional_labels = additional_labels

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

        if self.additional_labels is not None:
            for key in self.additional_labels:
                label = np.float(ExtractKeyFromFile(self.filenames[index], key))
                Tracker = np.append(Tracker, label)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            Tracker = self.target_transform(Tracker)

        return image, Tracker

    def get_corrupted_indexes(self):
        '''
        simple function to return corrupted samples in the dataset
        :TODO: ADD CALLABLE CRITERION TO HAVE SOME FLEXIBILITY ON WHAT SAMPLES TO DISCARD
        :return: indices of corrupted samples
        '''
        # find corrupted indexes
        corrupted_idxs = []
        print("finding corrupted indeces...")
        for j in range(len(self.filenames)):
            Tracker = ExtractTrackerMatrixFromFile(self.filenames[j])
            # remove corrupted points
            if Tracker[0] == 0 and Tracker[1] == 0 and Tracker[2] == 0 and Tracker[3] == 1 and Tracker[4] == 0 and Tracker[5] == 0 and Tracker[6] == 0:
                corrupted_idxs.append(j)
        print("detected {} corrupted samples...".format(len(corrupted_idxs)))
        return corrupted_idxs if len(corrupted_idxs)>0 else None

    def remove_corrupted_indexes(self, idxs):
        '''
        simple function to remove corrupted indeces from a list of indices (found via 'get_corrupted_indeces()')
        :param idxs: list or np.ndarray, containins the indices of corrupted samples that will be discarded.
                               NOTE THAT INDICES ARE ASSUMED TO BE TAKEN FROM THE SORTED DATASET AND THEY WILL BE
                               DISCARDED FROM THE SORTED DATASET. (default : None)
                               Example:
                               idxs = dataset.get_corrupted_indices()
                               or
                               idxs = [0,4,8,12]
        '''
        #discard corrupted files if requested
        if idxs is not None:
            print("discarding {} corrupted indeces...".format(len(idxs)))
            self.filenames = [y for idx,y in enumerate(self.filenames) if idx not in idxs ]

    def apply_target_transform(self, transform, **kwargs):
        '''
        function to add a transform or composition of transforms to the target vector.

        :param transform : callable or tuple of callables
        :param kwargs: additional parameters for transform, if tuple of transforms passed than additional
                       parameters must be grouped in dictionaries (see example below).

        EXAMPLE:

        extra parameters for transforms arranged in dicts:
        first = {"means": np.array(means), "stds": np.array(stds)}
        second = {"in_range": ((-3.5,3.5),)*label.shape[-1], "out_range": (0,1)}

        -apply single transform:
                dataset.apply_target_transform(NormalizeElementWise, **first)

            which is equilvalent to:
                dataset.apply_target_transform(NormalizeElementWise, means = np.array(means), stds = np.array(stds))

        -apply multiple transforms
                dataset.apply_target_transform((NormalizeElementWise,RescaleToRangeElementWise), first, second)
        '''

        #if multiple transforms
        if isinstance(transform, tuple):
            transform_list = []
            #check that there are as many transforms as additional dictionaries of extra parameters
            assert len(transform) == len(kwargs), "ERROR: if tuple of transform is passed there must be as many dicts" \
                                                  " of additional parameters ({}) as of transforms ({}).".format(len(kwargs),len(transform))
            for idx,args in enumerate(kwargs):
                #check inputs parameters have been passed in the right form
                assert isinstance(kwargs[args], dict), "ERROR: input parameters 'transform' is a tuple of transforms," \
                                                  " but additional parameters ({}) are not an instance of dict!".format(kwargs[args])

                #append the transformation, the try and exept handles if wrong parameters are passed raising an error
                try:
                    transform_list.append(transform[idx](**kwargs[args]))
                except TypeError:
                    print("ERROR: input additional parameters {} are not compatible with input parameters of {}".format(
                        kwargs[args], transform[idx]))
            #store transform as a target transform
            self.target_transform = Compose(transform_list)

        #if single transform is inputted as parameter is not a tuple
        else:
            #same as above
            try:
                self.target_transform = transform(**kwargs)
            except TypeError:
                print("ERROR: input additional parameters {} are not compatible with input parameters of {}".format(kwargs, transform))

    def __len__(self):
        return len(self.filenames)

    def get_filenames(self):
        return self.filenames

    def get_root(self):
        return self.root

    def get_mode(self):
        return self.mode

    def get_labels(self):
        return self.labels