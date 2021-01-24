import os
import numpy as np
import h5py
from skimage.io import imread
from datetime import datetime

from utils import preprocess_volume, preprocess_mask

class BrainMRIDataset():

    IMG_SHAPE = (256, 256)
    CHANNELS = ["pre-contrast", "FLAIR", "post-contrast"]

    def __init__(self, path='data/brainMRI.h5'):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    @staticmethod
    def make_dataset(raw_data_dir='./kaggle_3m', data_dir='./data'):

        if not os.path.exists(raw_data_dir):
            print('{} does not exist.\n You can download raw data at https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation')
            raise OSError
        
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
            
        patient_dirs = [
            d
            for d in os.listdir(raw_data_dir)
            if os.path.isdir(
                os.path.join(raw_data_dir, d)
            )
        ]

        for patient_dir in patient_dirs:
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
            images = np.empty((n_slices, *IMG_SHAPE, len(CHANNELS)), dtype=np.uint8)
            masks = np.empty((n_slices, *IMG_SHAPE), dtype=np.uint8)
            for i, name in enumerate(img_names):
                img_path = os.path.join(dir_path, name)
                prefix, ext = os.path.splitext(img_path)
                mask_path = prefix + '_mask' + ext
                
                images[i] = imread(img_path)
                masks[i] = imread(mask_path)
            
            images = preprocess_volume(images)
            masks = preprocess_mask(masks)
            patient = np.array(("_".join(patient_dir.split("_")[:-1]),)*n_slices)
            slices = np.array([
                int(x.split('.')[0].split("_")[4])
                for x in img_names
            ], dtype=np.uint8)

            sub_dir = os.path.join(data_dir, 'h5py')
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)

            h5_file_path = os.path.join(sub_dir, patient_dir + '.h5')
            with h5py.File(h5_file_path, 'w') as h5_file:
                h5_file.attrs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                h5_file.attrs['info'] = h5py.version.info
                h5_file.create_dataset("images", data=images)
                h5_file.create_dataset("masks", data=masks)
                h5_file.create_dataset("patients", data=patient.astype(h5py.string_dtype(encoding='utf-8')))
                h5_file.create_dataset("slices", data=slices)