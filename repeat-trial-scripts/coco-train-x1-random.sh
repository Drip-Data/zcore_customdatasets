CUDA_VISIBLE_DEVICES="2,3" python train_coreset_model.py --prune_rate 0.0001 --dataset coco --data_dir ./data --num_workers 10 --device cuda --score_file ./results/coco/zcore-coco-clip-resnet18-1000Ks-2sd-ri-1000nn-4ex-5/score.npy
CUDA_VISIBLE_DEVICES="2,3" python train_coreset_model.py --prune_rate 0.3 --dataset coco --data_dir ./data --num_workers 10 --device cuda --score_file ./results/coco/zcore-coco-clip-resnet18-1000Ks-2sd-ri-1000nn-4ex-5/score.npy
CUDA_VISIBLE_DEVICES="2,3" python train_coreset_model.py --prune_rate 0.5 --dataset coco --data_dir ./data --num_workers 10 --device cuda --score_file ./results/coco/zcore-coco-clip-resnet18-1000Ks-2sd-ri-1000nn-4ex-5/score.npy
CUDA_VISIBLE_DEVICES="2,3" python train_coreset_model.py --prune_rate 0.7 --batch_size 64 --dataset coco --data_dir ./data --num_workers 10 --device cuda --score_file ./results/coco/zcore-coco-clip-resnet18-1000Ks-2sd-ri-1000nn-4ex-5/score.npy
CUDA_VISIBLE_DEVICES="2,3" python train_coreset_model.py --prune_rate 0.9 --batch_size 32 --decay 0.0007 --dataset coco --data_dir ./data --num_workers 10 --device cuda --score_file ./results/coco/zcore-coco-clip-resnet18-1000Ks-2sd-ri-1000nn-4ex-5/score.npy

