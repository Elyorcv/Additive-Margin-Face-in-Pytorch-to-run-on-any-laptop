import numpy as np
from torchvision.datasets import ImageFolder

class TripDataset(ImageFolder):
    
    def __init__(self, file_path, transform=None):

        super(TripDataset, self).__init__(file_path,transform)
        self.idx_to_class = self.get_idx_to_class(self.class_to_idx)
        self.idx_path_dict,self.idx_imgs_count = self.get_idx_path_dict(self.imgs)
        self.num_classes = len(self.idx_path_dict)
        
    def get_idx_to_class(self,class_to_idx):
        idx_to_class = {}
        for k,v in class_to_idx.items():
            idx_to_class[v] = k
        return idx_to_class
    
    def get_idx_path_dict(self,imgs):
        idx_path_dict = {}
        idx_imgs_count = {}
        for path,idx in imgs:
            if not idx in idx_path_dict.keys():
                idx_path_dict[idx] = [path]
                idx_imgs_count[idx] = 1
            else:
                idx_path_dict[idx].append(path)
                idx_imgs_count[idx] += 1
        return idx_path_dict,idx_imgs_count
    
    def __getitem__(self, idx):
        np.random.seed()
        
        assert idx >= 0 and idx < self.num_classes, 'index over_range'
        
        def transform(img_path):
            img = self.loader(img_path)
            return self.transform(img)
        
        [pos1, pos2] = np.random.choice(range(self.idx_imgs_count[idx]),2).tolist()
        pos1, pos2 = self.idx_path_dict[idx][pos1], self.idx_path_dict[idx][pos2]
        others = list(range(self.num_classes))
        others.remove(idx)
        others = np.random.permutation(others)
        neg_class = np.random.choice(others,1)[0]
        neg = np.random.choice(range(self.idx_imgs_count[neg_class]),1)[0]
        neg = self.idx_path_dict[neg_class][neg]
        return transform(pos1),transform(pos2),transform(neg)

    def __len__(self):
        return self.num_classes