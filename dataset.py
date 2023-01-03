import os
import random
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class CO3D(Dataset):
    def __init__(self, configs, train=True, transform=None):
        self.configs = configs
        self.transform = transform

        if self.configs.dataset.use_all_categories:
            categories = os.listdir(self.configs.dataset.root)
            self.root = os.path.join(self.configs.dataset.root)
            # Not Implemented
        else:
            self.root = os.path.join(self.configs.dataset.root, self.configs.dataset.category_name)

        with open(os.path.join(self.root, 'frame_annotations.json')) as f: 
            frame_annotations = json.load(f)# [annotation for annotation in os.listdir(self.root) if annotation.split('.')[-1] == 'json']

        per_sequence_frame_annotations = {}
        for frame_annotation in frame_annotations:
            sequence_name = frame_annotation['sequence_name']
            if sequence_name in per_sequence_frame_annotations.keys():
                per_sequence_frame_annotations[sequence_name].append(frame_annotation)
            else:
                per_sequence_frame_annotations[sequence_name] = [frame_annotation]
        
        self.images_sets = []
        self.poses_sets = []

        for _, frame_annotations in per_sequence_frame_annotations.items():
            for _ in range(self.configs.dataset.n_sample_per_sequence):
                sequence_size = len(frame_annotations)
                random_set = self.getrandomset(sequence_size)
                
                n_nearby_frame_annotations = [frame_annotations[i] for i in random_set]

                images_set = [annotation['image']['path'] for annotation in n_nearby_frame_annotations]
                poses_set = [annotation['viewpoint'] for annotation in n_nearby_frame_annotations]

                self.images_sets.append(images_set)
                self.poses_sets.append(poses_set)

    def getrandomset(self, sequence_size):
        random_set = []
        for _ in range(self.configs.dataset.n_nearby_inputs + 1):
            n = random.randint(0, sequence_size - 1)
            while n in random_set:
                n = random.randint(0, sequence_size - 1)
            random_set.append(n)
        return random_set

    def __len__(self):
        return len(self.poses_sets)

    def __getitem__(self, index):
        images_set = self.images_sets[index]
        poses_set = self.poses_sets[index]

        images = []
        poses = []
        for image_path, pose in zip(images_set, poses_set):
            image_path = os.path.join(self.configs.dataset.root, image_path)
            image = Image.open(image_path)
            image = self.transform(image)
            images.append(image)
            
            R = torch.tensor(pose['R'], dtype=torch.float)
            T = torch.tensor(pose['T'], dtype=torch.float)

            pose = torch.cat([R, T.reshape(3, 1)], dim=1)
            pose = torch.cat([pose, torch.tensor([[0, 0, 0, 1]], dtype=torch.float)])
            poses.append(pose)
        
        images = torch.stack(images)
        poses = torch.stack(poses)
        return images, poses
