import os.path
import pathlib
import shutil

import cv2
import pandas as pd
import urllib.request
import pytorch_lightning
from sklearn.model_selection import train_test_split
import torch
from torchvision.transforms import RandomHorizontalFlip, Compose
from torchvision.transforms import transforms
from tqdm import tqdm

from lib.datasets import DatasetSplit
from lib.datasets.label_extractor import extract_labels
from lib.datasets.labeled_dataset import LabeledDataset
from lib.evaluator import eval_classifier
from lib.models.generic_classifier import GenericClassifier
from lib.models.pix2pix import Pix2Pix
from lib.trainer import setup_torch_device, setup
from lib.transform import transform, ZeroPaddingResize
from lib.transform.face_crop_transformer import FaceCrop
from lib.transform.face_segmentation_transformer import FaceSegmentation
from lib.transform.facial_landmarks_478_transformer import FacialLandmarks478
from lib.transform.pix2pix_transformer import Pix2PixTransformer
from lib.utils import glob_dir, get_last_ckpt, move_files

SEED = 42

class GANApproach_1:
    def preprocess(input_path: str, img_size: int = 512, align: bool = True, test_size: float = 0.1,
                shuffle: bool = True, output_dir: str = None, num_workers: int = 8):
        """
        Run all pre-processing steps (split, crop, segmentation, landmark) at once.
        """
        print(f"Starting preprocessing with: input_path={input_path}, img_size={img_size}, align={align}")
        
        if output_dir is None:
            output_dir = input_path
        output_dir = os.path.join(output_dir, 'original')

        # Split dataset
        split_file = extract_labels(input_path)
        if split_file:
            df = pd.read_csv(split_file)
            for split, folder in enumerate(['train', 'val', 'test']):
                move_files([os.path.join(input_path, img_path.replace('\\', os.sep)) for img_path in
                            df[df['split'] == split]['image_path'].values],
                        os.path.join(output_dir, folder))
            shutil.copyfile(split_file, os.path.join(output_dir, pathlib.Path(split_file).name))
        else:
            files = glob_dir(os.path.join(input_path))
            print(f'Found {len(files)} images')
            train_files, test_files = train_test_split(files, test_size=test_size, shuffle=shuffle)
            move_files(train_files, os.path.join(output_dir, 'train'))
            move_files(test_files, os.path.join(output_dir, 'val'))

        # Apply Transformers
        output_dir = transform(output_dir, img_size, False, FaceCrop(align),
                            num_workers=num_workers)
        output_dir = transform(output_dir, img_size, False, FaceSegmentation(),
                            num_workers=num_workers)
        output_dir = transform(output_dir, img_size, True, FacialLandmarks478(),
                            num_workers=num_workers)


    def train_pix2pix(data_dir: str, log_dir: str, models_dir: str, output_dir: str, dataset_name: str,
                    epoch: int = 0, n_epochs: int = 200, batch_size: int = 1, lr: float = 0.0002,
                    b1: float = 0.5, b2: float = 0.999, n_cpu: int = 2, img_size: int = 256,
                    checkpoint_interval: int = 500, device: int = 0):
        """
        Train the pix2pix GAN for generating faces based on given landmarks.
        """
        print(f"Starting pix2pix training with: dataset_name={dataset_name}, n_epochs={n_epochs}")
        
        setup_torch_device(device, SEED)
        ckpt_file = get_last_ckpt(models_dir)
        resume_ckpt = None
        out_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        if ckpt_file is not None:
            resume_ckpt = os.path.join(models_dir, ckpt_file)
            model = Pix2Pix.load_from_checkpoint(checkpoint_path=resume_ckpt)
        else:
            model = Pix2Pix(data_dir, models_dir, out_dir, n_epochs, dataset_name, batch_size, lr, b1,
                            b2, n_cpu, img_size, device)
        trainer = setup(model, log_dir, models_dir, n_epochs, device,
                        checkpoint_interval=checkpoint_interval)
        trainer.fit(model, ckpt_path=resume_ckpt)


    def train_classifier(data_dir: str, num_classes: int, learning_rate: float = 0.0003,
                        batch_size: int = 128, n_epochs: int = 100, device: int = 0,
                        output_dir: str = 'output', monitor: str = 'val_loss',
                        metric_mode: str = 'min', save_top_k: int = 1, early_stop_n: int = 5,
                        num_workers: int = 8):
        """
        Run the training.
        """
        print(f"Starting classifier training with: data_dir={data_dir}, n_epochs={n_epochs}")
        
        pytorch_lightning.seed_everything(SEED, workers=True)
        train_db = LabeledDataset(data_dir, DatasetSplit.TRAIN, Compose([
            GenericClassifier.weights.transforms(),
            RandomHorizontalFlip(),
        ]))
        val_db = LabeledDataset(data_dir, DatasetSplit.VALIDATION,
                                Compose([GenericClassifier.weights.transforms()]))
        model = GenericClassifier(train_db=train_db, val_db=val_db, multi_label=train_db.is_multi_label,
                                batch_size=batch_size, learning_rate=learning_rate,
                                num_workers=num_workers,
                                device=setup_torch_device(device, SEED), classes=train_db.classes,
                                class_weights=train_db.class_weight)
        models_dir = os.path.join(output_dir, 'models')
        trainer = setup(model=model, log_dir=os.path.join(output_dir, 'logs'),
                        models_dir=models_dir, num_epoch=n_epochs,
                        device=device, monitor=monitor, metric_mode=metric_mode,
                        save_top_k=save_top_k, early_stop_n=early_stop_n)
        trainer.fit(model)
        eval_classifier(models_dir, data_dir, batch_size, device, output_dir, num_workers)


    def anonymize_image(model_file: str, input_file: str, output_file: str, img_size: int = 512,
                        align: bool = True, device: int = 0):
        """
        Anonymize one face in a single image.
        """
        img = cv2.imread(input_file)
        if img is not None:
            faces = FaceCrop(align)(img)
            for idx, f in enumerate(faces):
                img = f
                img = ZeroPaddingResize(img_size)(img)
                img = FacialLandmarks478()(img)
                img = Pix2PixTransformer(model_file, img_size, device)(img)
                img = torch.squeeze(img)
                img = (img + 1) / 2
                pil_img = transforms.ToPILImage()(img)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                pil_img.save(os.path.join(os.path.dirname(output_file), 
                                        f'{idx}_{os.path.basename(output_file)}'))


    def anonymize_directory(self, model_file: str, input_directory: str, output_directory: str,
                            img_size: int = 512, align: bool = True, device: int = 0):
        """
        Anonymize a set of images in a directory.
        """
        for file in tqdm(os.listdir(input_directory), desc=f"Anonymizing from {input_directory}"):
            input_file = os.path.join(input_directory, file)
            output_file = os.path.join(output_directory, os.path.basename(file))
            self.anonymize_image(model_file, input_file, output_file, img_size, align, device)

