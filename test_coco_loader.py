# 测试COCO数据集加载器
# 也可用来测试自定义数据集加载器

import os
import torch
import torchvision.transforms as transforms
from core.train.pruned_dataset import CocoSingleLabelDataset

def test_coco_loader():
    # Set paths
    data_dir = "data/coco"
    train_json = os.path.join(data_dir, "train_single_class.json")
    train_img_dir = os.path.join(data_dir, "train2014")
    
    print(f"JSON path: {os.path.abspath(train_json)}")
    print(f"Image dir: {os.path.abspath(train_img_dir)}")
    
    # Verify paths exist
    if not os.path.exists(train_json):
        print(f"ERROR: JSON file not found: {train_json}")
        return False
    
    if not os.path.exists(train_img_dir):
        print(f"ERROR: Image directory not found: {train_img_dir}")
        return False
    
    # Create simple transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Try to load dataset
    print("Loading dataset...")
    try:
        dataset = CocoSingleLabelDataset(train_img_dir, train_json, transform)
        print(f"Success! Dataset contains {len(dataset)} images")
        print(f"Available classes: {dataset.classes}")
        
        # Test accessing an item
        if len(dataset) > 0:
            image, label = dataset[0]
            print(f"First image shape: {image.shape}")
            print(f"First image label: {label}")
            
        return True
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting test...")
    test_result = test_coco_loader()
    print(f"Test completed with result: {'SUCCESS' if test_result else 'FAILURE'}") 