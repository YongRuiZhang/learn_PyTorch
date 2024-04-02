# 训练套路总结

1. 先看文件 [train.py](./train.py) 其中包含的步骤为：
   1. 获取数据（dataset），这一步同时获取训练集和测试集数据，然后分别获取size
   2. 使用DataLoader加载数据（指定BatchSize和Shuffle）
   3. 搭建神经网络模型结构，这一步可以直接写在文件中，但是根据经验，一般都是会新建一个文件，这里就新建了文件 [model.py](./model.py)
   4. 实例化模型
   5. 定义损失函数
   6. 定义优化器方法
   7. 设置指定参数，例如epoch还；有一些评价指标相关的参数，例如total_train_step，total_test_step
   8. 开始学习，外层循环为`for i in range(epoch)`,内层循环分成两步 
      1. 训练，这里可以先使用`modelName.train()`设置为训练模式，但是只有在模型中有BN或Dropout时有提升 
         1. 训练中先获取DataLoader中的images和targets，然后将images作为input放到模型中计算出结果，然后利用损失函数计算出损失。
         2. 然后利用优化器和方向传播对参数进行调优。首先要先使用`optimizerName.zero_grad()`将上一次的梯度调整为零，然后利用`lossName.backward()`进行方向传播计算梯度，最后利用`optimizerName.step()`对参数调优
      2. 测试，这里可以先使用`modelName.test()`设置为测试模式，但是只有在模型中有BN或Dropout时有提升。测试步骤对于训练没有用处，只是计算一些评价指标的作用
         1. 测试要在没有梯度的环境中进行，因此使用`with torch.no_grad():`
         2. 然后获取DataLoader中的images和targets，然后将images作为input放到模型中计算，得出概率结果。
         3. 根据概率结果得出预测结果（与targets的对应关系），可以考虑使用`outputs.argmax(1)`，然后就可以对结果进行计算一些评价指标，例如Accuracy、Precision、ReCall等
   9. 可以选择 使用TensorBoard对训练过程中的评价指标进行可视化
   10. 保存模型
2. 再看文件 [train_gpu_01.py](./train_gpu_01.py)

   这个文件介绍了第一种利用GPU训练的方式，但是这种方式只能应用到cuda上。因为是对**模型、损失函数、数据**调用`.cuda()`方法。
3. 再看文件 [train_gpu_02.py](./train_gpu_02.py)

   这个文件介绍了第二种利用GPU训练的方式。这种方式需要先设置使用设备，语句类似于：`device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")`,cuda:1表示第一块显卡，cuda:0和cuda是相同的。可以直接设置为"cpu"
   
   cuda是nvidia的GPU后端，MacBook M系列芯片可以用mps，写成：`device = torch.device("mps" if torch.has_mps else "cpu")`

   设置好设备变量后，仍然对模型、损失函数、数据调用`.to(device)`，即可转移到设备上
4. 再看文件 [test.py](./test.py)

   主要介绍了如何测试训练好后的模型。其实也是实际应用的一种，例如分类，预测。
   
   步骤就是先读入图片（一般就是PIL类型），然后利用`torchvision.transformers.Compose()`进行复合的transformers操作，一般需要先Resize为模型需要的图片size，然后再转换为Tensor类型

   注意：最好在转换前，先对PIL格式的图片调用`.convert('RGB')`转换为3通道，这样测试png和jpg格式的图片都不会有问题。

   然后加载网络模型，要对应第一种或是第二种加载方式。由于第一种加载方式会将模型结构一起加载过来，当模型的训练环境和本地测试的环境不同（一个是cuda另一个是cpu）时，就会报错，要在load函数中添加一个参数:`map_location = torch.device('cpu)`,与本地环境对应即可

   之后就是测试使用了，一般需要先调用`.eval()`转换为评估模式，然后在`with torch.no_grad():`无梯度环境中计算预测结果。最终操作计算结果即可