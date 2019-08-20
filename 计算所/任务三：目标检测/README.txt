目标识别任务介绍：1.调研faster rcnn，并了解网络的输入格式，以及如何获取输入。 2.生成数据集，每份样本是在大图片中随机放了一些小数字，要求小数字尺度放大0.5到2倍，旋转-15到15度且有一定间距。 3.代码实现，训练网络将小数字框出来。另外还可以尝试将小数字放大5倍且有重叠，再训练网络识别，作为测试集。

文件介绍：

任务报告.ipynb ：本次任务报告

fig：报告图例

datasets ：原小数字的数据

generater.ipynb ：生成标准数据集的代码

目标检测数据集1.1.rar：数据集的压缩包备份

Faster R-CNN调研报告.ipynb：调研报告

fig：调研报告图例

网络代码和模型都在main文件夹里:

model : 文件夹内是保存的模型参数和训练时的损失文件

train_image，validation_image，test_image  分别是对应数据集的训练图片

train_info_c.txt，validation_info_c.txt，test_info.txt 分别是对应数据集的标注文件

train_check，validation_check，test_check 分别是对应数据集图片被正确标记的结果

predict_image 是验证集的所有预测结果

test_predict1 是测试集被第一个模型的预测结果

test_predict2 是测试集被第二个模型的预测结果