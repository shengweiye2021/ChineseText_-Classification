# ChineseText_-Classification

运行方式： 1,cd到该项目的路径下；

2,选择运行方式 ：
选择单机单卡gpu： python run.py
选择单机单线程多gpu：python run.py 
选择单机多线程多gpu:python -m torch.distributed.launch --nproc_per_node 4 --master_port 8005 run.py --device_ids=0,1,2,3

选择哪种运行方式就在run.py文件将相应的注释消掉，并将其他运行方式注释； 在使用多gpu的情况下，尽量使用可以修改gpu的使用张数，并不是越多gpu越好； 多线程的情况，4是线程数，ids是指定的gpu，两个的数量是一样的。

默认运行的模型是transformer;

训练数据集有1300万条，epoch=20，batch_size=128(可以设置为1024)

[数据集](链接：https://pan.baidu.com/s/1Lc9dKoXHFJK8c0yYxvZTMA)提取码：0pfx 

