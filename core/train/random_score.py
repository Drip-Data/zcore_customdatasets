# 根据score.npy文件生成随机分数 做对比实验
import numpy as np

data = np.load("/data1/anquan/D/zcore/results/coco/zcore-coco-clip-resnet18-1000Ks-2sd-ri-1000nn-4ex-4/score.npy")
# 排序data
data = np.sort(data)
# 随机打印10个元素
print(data[np.random.randint(0, len(data), 10)])

# 打印形状
print(data.shape[0])

# # 生成data2，data和data形状相同，data2的元素是0-1之间的随机数
# data2 = np.random.rand(data.shape[0])
# # 将data2 保存为和score.npy文件相同的文件
# np.save("/data1/anquan/D/zcore/results/coco/zcore-coco-clip-resnet18-1000Ks-2sd-ri-1000nn-4ex-1/score2.npy", data2)








