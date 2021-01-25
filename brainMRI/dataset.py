import os
import numpy as np
import h5py
from skimage.io import imread
from datetime import datetime

from .utils import preprocess_volume, preprocess_mask

class Dataset():
    """
    TCGA-LGG dataset of brain MRIs for Lower Grade Glioma segmentation.

    Attributes:
        IMG_SHAPE : tuple
            Shape of the images: (H, W).
        VOLUMES : list
            Names of the volumes.

    Methods:
        make_dataset(raw_data_dir='./kaggle_3m', data_dir='./data')
            Creates a virtual HDF5 dataset with preprocessed images and metadata.
    """

    IMG_SHAPE = (256, 256)
    VOLUMES = ["pre-contrast", "FLAIR", "post-contrast"]

    def __init__(self, path="./data/brainMRI.h5", train=True, volume=None):
        """Initializes the brain MRI dataset.
            
        Args:
            path : str, optional
                Path to the virtual HDF5 dataset.
                Default is './data/brainMRI.h5'.
            train : bool, optional
                Slice of the dataset to select.
                Default is True, selects 80% of the patients.
            volume : str, optional.
                Volume images to return.
                Default is None, returns all the volumes.

        Returns: instance of the brain MRI dataset
        """

        if not os.path.exists(path):
            raise RuntimeError("Dataset not found at '{}'.\nUse Dataset.make_dataset() to create it.".format(path))

        assert volume in [None,] + Dataset.VOLUMES, 'volume can only be None or one of {}'.format(Dataset.VOLUMES)

        self.dataset = h5py.File(path, 'r')
        self.volume = volume
        
        self.index_vector = np.arange(self.dataset['images'].shape[0])
        patients = np.unique(self.patients)
        data_patients = np.random.choice(
            patients, 
            size=int((0.8 if train else 0.2) * len(patients)), 
            replace=False
        )
        bool_vector = np.zeros_like(self.index_vector)
        for patient in data_patients:
            bool_vector = bool_vector + np.array(self.patients == patient)
        self.index_vector = np.where(bool_vector)[0]

    def __getitem__(self, index):
        index = self.index_vector[index]

        img = self.dataset['images'][index].astype(np.float32)
        mask = self.dataset['masks'][index]
        patient = self.dataset['patients'][index]
        slice = self.dataset['slices'][index]

        if self.volume is not None:
            img = img[..., self.VOLUMES.index(self.volume)]
            img = img[..., np.newaxis]

        return img, mask, (patient, slice)

    def __len__(self):
        return len(self.index_vector)

    @property
    def images(self):
        return self.dataset["images"][self.index_vector]

    @property
    def masks(self):
        return self.dataset["masks"][self.index_vector]

    @property
    def patients(self):
        return self.dataset["patients"][self.index_vector]

    @property
    def slices(self):
        return self.dataset["slices"][self.index_vector]

    @staticmethod
    def make_dataset(raw_data_dir='./kaggle_3m', data_dir='./data'):
        """Creates a virtual HDF5 dataset with preprocessed images and metadata.
        
        Data should be previously downloaded from https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation.

        Args:
            raw_data_dir : str, optional
                Path to the raw data directory.
            data_dir : str, optional
                Path to the processed data directory.
        """

        # check data directories
        if not os.path.exists(raw_data_dir):
            print('{} does not exist.\n You can download raw data at https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation'.format(
                raw_data_dir
            ))
            raise OSError
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        h5_dir = os.path.join(data_dir, 'hdf5')
        if not os.path.exists(h5_dir):
            os.mkdir(h5_dir)
            
        patient_dirs = [
            d
            for d in os.listdir(raw_data_dir)
            if os.path.isdir(
                os.path.join(raw_data_dir, d)
            )
        ]

        n_samples = 0
        for patient_dir in patient_dirs:
            
            # retrieve patient images and masks
            dir_path = os.path.join(raw_data_dir, patient_dir)
            img_names = [
                x
                for x in os.listdir(dir_path)
                if 'mask' not in x
            ]
            img_names.sort(
                key=lambda x: int(x.split(".")[0].split("_")[4])
            )
            n_slices = len(img_names)
            n_samples += n_slices
            images = np.empty((n_slices, *Dataset.IMG_SHAPE, len(Dataset.VOLUMES)), dtype=np.uint8)
            masks = np.empty((n_slices, *Dataset.IMG_SHAPE), dtype=np.uint8)
            for i, name in enumerate(img_names):
                img_path = os.path.join(dir_path, name)
                prefix, ext = os.path.splitext(img_path)
                mask_path = prefix + '_mask' + ext
                
                images[i] = imread(img_path)
                masks[i] = imread(mask_path)
            
            # preprocess images and metadata
            images = preprocess_volume(images)
            masks = preprocess_mask(masks)
            patient = np.array(("_".join(patient_dir.split("_")[:-1]),)*n_slices)
            slices = np.array([
                int(x.split('.')[0].split("_")[4])
                for x in img_names
            ], dtype=np.uint8)

            # create patient dataset
            h5_file_path = os.path.join(h5_dir, patient_dir + '.h5')
            with h5py.File(h5_file_path, 'w') as h5_file:
                h5_file.attrs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                h5_file.attrs['info'] = h5py.version.info
                h5_file.create_dataset("images", data=images)
                h5_file.create_dataset("masks", data=masks)
                h5_file.create_dataset("patients", data=patient.astype(h5py.string_dtype(encoding='utf-8')))
                h5_file.create_dataset("slices", data=slices)

        # create virtual layouts
        layouts = {
            "images": h5py.VirtualLayout(
                shape=(n_samples, *Dataset.IMG_SHAPE, len(Dataset.VOLUMES)), 
                dtype=np.float16
                ),
            "masks": h5py.VirtualLayout(
                shape=(n_samples, *Dataset.IMG_SHAPE), 
                dtype=np.uint8
                ),
            "patients": h5py.VirtualLayout(
                shape=(n_samples,), 
                dtype=h5py.string_dtype(encoding='utf-8')
                ),
            "slices": h5py.VirtualLayout(
                shape=(n_samples,), 
                dtype=np.uint8
                )
        }
        
        # fill the virtual layouts
        i = 0
        for filename in os.listdir(h5_dir):
            file_path = os.path.join(h5_dir, filename)
            with h5py.File(file_path, "r") as h5_file:
                n_slices = h5_file['slices'].shape[0]
                for k in h5_file.keys():
                    layouts[k][i:i+n_slices] = h5py.VirtualSource(h5_file[k])
                i += n_slices
        
        # create virtual dataset
        vds_path = os.path.join(data_dir, 'brainMRI.h5')
        with h5py.File(vds_path, "w") as h5_file:
            h5_file.attrs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            h5_file.attrs['h5py_info'] = h5py.version.info
            h5_file.attrs['dataset'] = 'TCGA-LGG Brain MRI'
            h5_file.attrs['github'] = 'https://github.com/giacomodeodato/BrainMRIDataset'
            h5_file.attrs['website'] = 'https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation'
            for name, layout in layouts.items():
                h5_file.create_virtual_dataset(name, layout)