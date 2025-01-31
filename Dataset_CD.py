
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from patchify import patchify
import numpy as np
from Utils import seeding


class change_detection_dataset(Dataset):
    def __init__(self, root_path, patch_size) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pre_change_path = os.path.join(root_path,"A")
        self.post_change_path = os.path.join(root_path,"B")
        self.change_label_path = os.path.join(root_path,"label")
        self.fname_list=os.listdir(self.pre_change_path)

    def __getitem__(self, index):
        fname = self.fname_list[index]
        pre_img = Image.open(os.path.join(self.pre_change_path,fname)).convert("RGB")
        post_img = Image.open(os.path.join(self.post_change_path,fname)).convert("RGB")
        change_label = Image.open(os.path.join(self.change_label_path,fname)).convert("1")
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])

        pre_tensor = transform(pre_img)
        post_tensor = transform(post_img)
        label_tensor = transform(change_label)

        pre_tensor_patches = self.create_patches(pre_tensor )                               ## we only patch the images , but labels are NOT patched
        post_tensor_patches = self.create_patches( post_tensor)

        # return {'pre':pre_tensor_patches,'post':post_tensor_patches,'label':label_tensor,'fname':fname}
        return pre_tensor_patches, post_tensor_patches, label_tensor, fname

    def create_patches(self, image):
        '''create image patches'''
        image_np = image.permute(1, 2, 0).numpy()                                                                       # Convert to (H, W, C)
        patches = patchify(image_np, (self.patch_size, self.patch_size, image_np.shape[2]), step=self.patch_size)
        patches = patches.reshape(-1, self.patch_size, self.patch_size, image_np.shape[2])                              # Flatten patches
        patches = torch.tensor(patches).permute(0, 3, 1, 2)                                                             # Convert to (num_patches, C, patch_size, patch_size)
        patches = patches.reshape(256, 768)

        return patches


    def __len__(self):
        return len(self.fname_list)
    

if __name__ == "__main__":

    seeding(42)

    patch_size = 16
    train_path="D://Datasets//Levir_croped_256//LEVIR_CD//train"
    dataset = change_detection_dataset(train_path, patch_size )

# # Iterate through the dataset
    for x, y, l, n in dataset:
        print(x.shape)  # 256, 768
        print(y.shape)  # 256, 768
        print(l.shape)  # 1, 256, 256
        print(n)
        print (x.device)
        print (y.device)
        print ("x type", x.dtype)   # torch.float32
        print ("y type", y.dtype)   # torch.float32
        print ("l type", l.dtype)   # torch.float32

        print(f"x Min value: {x.min()}, x Max value: {x.max()}")   # Min value: 0.0, x Max value: 1.0
        print(f"y Min value: {y.min()}, y Max value: {y.max()}")   # Min value: 0.0, y Max value: 1.0
        print(f"l Min value: {l.min()}, l Max value: {l.max()}")   # Min value: 0.0, l Max value: 1.0
        
        break