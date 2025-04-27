import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
from PIL import Image
from torch.utils.data import Dataset

# 单一标签的COCO数据集—— json文件导入
class CocoSingleLabelDataset(Dataset):
    def __init__(self, img_dir, annotation_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Load annotation file (JSON format)
        with open(annotation_file, 'r') as f:
            annotations_list = json.load(f)
        
        # Convert list format to our internal representation
        self.annotations = {}
        self.img_filenames = []
        all_categories = set()
        
        for item in annotations_list:
            filename = item["file_name"]
            # Take the first category as the single label
            if len(item["categories"]) > 0:
                category = item["categories"][0]
                self.annotations[filename] = category
                self.img_filenames.append(filename)
                all_categories.add(category)
        
        # 将类别名称映射为索引
        self.classes = sorted(list(all_categories))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.img_filenames)
    
    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 获取标签并转换为索引
        label = self.annotations[img_filename]
        
        # 如果标签是列表，则返回一个包含标签索引和权重的元组
        if isinstance(label, list):
            # 返回一个包含标签索引和权重的元组
            label_idx, weight = label
            return image, (torch.tensor(label_idx), torch.tensor(weight))
        else:
            # 返回一个包含标签索引的元组
            label_idx = self.class_to_idx[label]
            return image, label_idx
    
    @property
    def samples(self):
        """Return a list of tuples (path, target) to mimic ImageFolder structure"""
        samples = []
        for img_filename in self.img_filenames:
            img_path = os.path.join(self.img_dir, img_filename)
            label = self.annotations[img_filename]
            if isinstance(label, list):
                label_idx = label[0]  # 获取标签索引
            else:
                label_idx = self.class_to_idx[label]
            samples.append((img_path, label_idx))
        return samples

def load_coreset_dataset(args):
    print(f"Loading {args.prune_rate} pruned {args.dataset}.")
   
    # Load dataset.
    if "cifar" in args.dataset: train_data, test_data = load_cifar(args)
    elif args.dataset == "imagenet": train_data, test_data = load_imagenet(args)
    elif "eurosat" in args.dataset: train_data, test_data = load_eurosat(args)
    elif args.dataset == "coco": train_data, test_data = load_coco(args)
    # elif args.dataset == "custom": train_data, test_data = load_custom(args)
    else: raise ValueError(f"{args.dataset} not recognized.")

    # Prune dataset.
    score = np.load(args.score_file)
   
    """
    # Convert for TDDS format score and mask.
    score_alt = (score - min(score)) / (max(score) - min(score))
    data_mask = np.load(args.score_file.replace("score.npy", "data_mask.npy"))
    score = np.zeros(n).astype(np.float32)
    for i in range(n): score[i] = score_alt[np.where(data_mask == i)]
    """
    
    if "cifar" in args.dataset:
        train_data.targets = [[t, score[i]] 
                              for i, t in enumerate(train_data.targets)]
    elif args.dataset == "coco" and hasattr(train_data, "img_filenames"):
        # 将标签索引和权重添加到训练数据中
        for i, filename in enumerate(train_data.img_filenames):
            # 将注释格式转换为包含分数权重
            category = train_data.annotations[filename]
            # 如果标签不是列表，则添加分数权重 （避免重复处理）
            if not isinstance(category, list):
                label_idx = train_data.class_to_idx[category] 
                train_data.annotations[filename] = [label_idx, score[i]]
    else:
        train_data.samples = [(s[0], [s[1], score[i]]) 
                              for i, s in enumerate(train_data.samples)]
    
    coreset_mask = np.argsort(score)[int(args.prune_rate * len(train_data)):]
    coreset = torch.utils.data.Subset(train_data, coreset_mask)

    # Dataset loaders.
    train_loader = torch.utils.data.DataLoader(
        coreset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader

def load_imagenet(args):

    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    path = os.path.join(args.data_dir, args.dataset, "ILSVRC", "Data", "CLS-LOC")
    train_data = datasets.ImageFolder(os.path.join(path, "train"), train_transform)
    test_data = datasets.ImageFolder(os.path.join(path, "val"), test_transform)

    return train_data, test_data

def load_eurosat(args):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # https://github.com/Rumeysakeskin/EuroSat-Satellite-CNN-and-ResNet
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    path = os.path.join(args.data_dir, args.dataset)
    train_data = datasets.ImageFolder(os.path.join(path,"train"), train_transform)
    test_data = datasets.ImageFolder(os.path.join(path,"val"), test_transform)

    return train_data, test_data

def load_cifar(args): 

    if args.dataset == "cifar10":
        mean = [0.4913725490196078, 0.4823529411764706, 0.4466666666666667] 
        std =  [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]
    elif args.dataset == "cifar100":
        mean = [0.5070588235294118, 0.48666666666666664, 0.4407843137254902]
        std = [0.26745098039215687, 0.2564705882352941, 0.27607843137254906]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    if args.dataset == "cifar10":   
        train_data = datasets.CIFAR10(args.data_dir, train=True, 
                                  transform=train_transform)
        test_data = datasets.CIFAR10(args.data_dir, train=False, 
                                 transform=test_transform)
    elif args.dataset == "cifar100":
        train_data = datasets.CIFAR100(args.data_dir, train=True, 
                                  transform=train_transform)
        test_data = datasets.CIFAR100(args.data_dir, train=False, 
                                 transform=test_transform)

    return train_data, test_data

def load_coco(args):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    path = os.path.join(args.data_dir, "coco")
    
    # 使用自定义数据集与JSON注释代替ImageFolder
    train_json = os.path.join(path, "train_single_class.json")
    test_json = os.path.join(path, "val_single_class.json")
    
    train_img_dir = os.path.join(path, "train2014")
    test_img_dir = os.path.join(path, "val2014")
    
    train_data = CocoSingleLabelDataset(train_img_dir, train_json, train_transform)
    test_data = CocoSingleLabelDataset(test_img_dir, test_json, test_transform)

    return train_data, test_data
