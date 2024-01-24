# 基于成员推理攻击的深度神经网络逐层隐私泄漏探究

深度神经网络在记忆训练数据信息时，容易受到各种推理攻击，比如最常见的成员推理攻击。本工作聚焦于复现 Nasr 等人所设计的白盒推理攻击来对深度学习模型进行逐层隐私分析，我们参考 Rezaei 等人对梯度张量提取七种梯度范数，以减少攻击模型的复杂度。我们使用 CIAR100 数据集分别在 CNN、AlexNet、ResNet 上进行实验，实验结果表明不同神经网络的不同层泄漏的隐私是有区别的，并且这种区别并非 Nasr 等人所说的越往后的层泄漏的隐私越多。相反，在我们的实验结论中，我们认为不同层的隐私泄漏将取决于两个重要因素：即层的位置和层的参数量。

## 环境配置

```commandline
conda create --name <env_name> --file requirements.txt
```

- 从 [这里](https://github.com/bearpaw/pytorch-classification)获取预训练参数. 请按照下面的结构放置好文件:

```commandline
+-- trained_models
|  +-- densenet-bc-L190-k40
|  |  +-- model_best.pth.tar
|  +-- resnext-8x64d
|  |  +-- model_best.pth.tar
|  +-- resnet-110
|  |  +-- model_best.pth.tar
|  +-- alexnet
|  |  +-- model_best.pth.tar

```

## 运行

```commandline
conda activate <env_name>
python ShapleyMetrics.py
```

## 代码版权说明
1. pytorch-classification，源自[pytorch-classification](https://github.com/bearpaw/pytorch-classification)，引用其目标模型和预训练参数。
2. Collect.py，源自[Quantify MI Leakage](https://github.com/ganeshdg95/Leveraging-Adversarial-Examples-to-Quantify-Membership-Information-Leakage)，引用其收集梯度、损失等信号。
3. ModelShokri.py，源自[Quantify MI Leakage](https://github.com/ganeshdg95/Leveraging-Adversarial-Examples-to-Quantify-Membership-Information-Leakage)引用其对Nasr等人攻击模型的搭建。
4. Collect_Gradient_Norm.py，自主实现，收集七种梯度范数作为信号。
5. Main.py，自主实现，代码运行的主函数。
6. ShapleyMetrics.py，自主实现，测量每一层的MIA隐私泄漏量。
7. utils.py，源自[Quantify MI Leakage](https://github.com/ganeshdg95/Leveraging-Adversarial-Examples-to-Quantify-Membership-Information-Leakage)，参考其部分工具函数的实现。