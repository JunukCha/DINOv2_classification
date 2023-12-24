import os.path as osp

from PIL import Image
import scipy.io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class Flower(Dataset):
    def __init__(self, mode="train", split="1", model="dinov2"):
        super().__init__()
        self.img_root = "data/Flower/jpg"
        mat = scipy.io.loadmat('data/Flower/datasplits.mat')
        
        if mode == "train":
            self.indices = mat[f"trn{split}"][0]
        elif mode == "test":
            self.indices = mat[f"tst{split}"][0]
        elif mode == "valid":
            self.indices = mat[f"val{split}"][0]
        else:
            raise Exception("Not allowed mode!!!")
        
        self.label_dict = {}
        for i in range(1, 1361):
            self.label_dict[i] = (i - 1) // 80 % 17
        
        if model == "dinov2":
            normalize = T.Normalize([0.5], [0.5])
        elif model == "resnet50":
            normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.transform = T.Compose([
                T.ToTensor(), 
                T.Resize(244), 
                T.CenterCrop(224), 
                normalize, 
            ])

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        data_indices = self.indices[index]
        img_path = osp.join(self.img_root, f"image_{data_indices:04d}.jpg")
        img = Image.open(img_path)
        img = self.transform(img)
        label = self.label_dict[data_indices]
        return img, label
    

def get_dataloader(batch_size=16, shuffle=True, num_workers=4, mode="train", split="1", model="dinov2"):
    dataset = Flower(mode=mode, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader