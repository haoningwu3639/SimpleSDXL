import torch
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Note: You should write a DataLoader suitable for your own Dataset!!!
class SimpleDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.image_dir = os.path.join(self.root, 'data')
        folders = sorted(os.listdir(self.image_dir))
        self.image_list = [os.path.join(self.image_dir, file) for file in folders]
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = self.image_list[index]
        text = image.split('/')[-1]
        prompt = text.replace('_', ' ')[:-4]
        image =  Image.open(image).convert('RGB')
        
        # haoning: TODO: remember to edit here
        original_sizes = [(1024, 1024)]
        image = image.resize((1024, 1024))
        crop_top_lefts = [(0, 0)]
        image = transforms.ToTensor()(image)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()

        # normalize
        image = image * 2. - 1.

        return {"image": image, "prompt": prompt, "original_sizes": original_sizes, "crop_top_lefts": crop_top_lefts}

if __name__ == '__main__':
    train_dataset = SimpleDataset(root="./")
    print(train_dataset.__len__())

    train_data = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=False)
    # B C H W
    for i, data in enumerate(train_data):
        print(i)
        print(data['image'].shape)
        print(data['prompt'])