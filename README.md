# Deep-Learning_HW2

This repository contains the implementation of the LeNet model using different approaches and compares their results. 

Table of Contents
+ LeNet-Basic
+ CustomMLP
+ LeNet-Regularization
+ LeNet-Data Augmentation
+ Conclusion

-------

## LeNet-Basic
<div align="center">
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/bd14d1a6-28ee-4596-89b0-2fa7db7dc8c2" width="500"/>
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/50d9ea03-0795-478d-9220-042acb90677b" width="500"/>
</div>


The basic LeNet model achieved the following performance metrics at Epoch 15:

Train Loss: 0.0069, Train Acc: 99.77%
Test Loss: 0.0540, Test Acc: 98.91%

<img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/3056a6ff-c1a3-4b67-aed5-ff28accab581" width="500"/>

The number of parameters for each layer is as follows:
1. **conv1:**
   - Input channels: 1
   - Kernel size: 5x5
   - Output channels: 6
   - Weight parameters: 5 * 5 * 1 * 6 = 150
   - Bias parameters: 6
   - Total parameters: 150 + 6 = 156

2. **conv2:**
   - Input channels: 6
   - Kernel size: 5x5
   - Output channels: 16
   - Weight parameters: 5 * 5 * 6 * 16 = 2400
   - Bias parameters: 16
   - Total parameters: 2400 + 16 = 2,416

3. **conv3:**
   - Input channels: 16
   - Kernel size: 4x4
   - Output channels: 120
   - Weight parameters: 4 * 4 * 16 * 120 = 30720
   - Bias parameters: 120
   - Total parameters: 30720 + 120 = 30,840

4. **fc1:**
   - Input size: 120
   - Output size: 84
   - Weight parameters: 120 * 84 = 10080
   - Bias parameters: 84
   - Total parameters: 10080 + 84 = 10,164

5. **fc2:**
   - Input size: 84
   - Output size: 10
   - Weight parameters: 84 * 10 = 840
   - Bias parameters: 10
   - Total parameters: 840 + 10 = 850

### Total Parameters
The total number of parameters in the LeNet-5 model is 44,426.

------

## CustomMLP
<div align="center">
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/b6274945-3874-4ad9-8fbd-93c9e23a0967" width="500"/>
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/b39a8880-9497-4340-ad63-66801a1fc84e" width="500"/>
</div>


The CustomMLP model achieved the following performance metrics at Epoch 15:

Train Loss: 0.0141, Train Acc: 99.53%
Test Loss: 0.1059, Test Acc: 97.57%

Compared to LeNet-Basic, the CustomMLP model exhibits slower convergence and slightly lower accuracy. This observation suggests that the CustomMLP architecture might require further optimization or adjustments.

<img src= "https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/145a21a0-b04e-4e1d-8bf3-3469823e21cf" width="500"/>

The number of parameters for each layer is as follows:
1. **fc1:**
   - Input size: 28 * 28 = 784
   - Output size: 60
   - Weight parameters: 784 * 60 = 47,040
   - Bias parameters: 60
   - Total parameters: 47,040 + 60 = 47,100

2. **fc2:**
   - Input size: 60
   - Output size: 30
   - Weight parameters: 60 * 30 = 1,800
   - Bias parameters: 30
   - Total parameters: 1,800 + 30 = 1,830

3. **fc3:**
   - Input size: 30
   - Output size: 10
   - Weight parameters: 30 * 10 = 300
   - Bias parameters: 10
   - Total parameters: 300 + 10 = 310

### Total Parameters
The total number of parameters in the CustomMLP model is 49,240.


------

## LeNet-Regularization

In this version of the LeNet model, regularization techniques such as Dropout and Batch Normalization were incorporated to improve performance and prevent overfitting.

Dropout: Randomly sets a fraction of input units to zero during training to prevent overfitting.

Batch Normalization: Normalizes the activations of each layer, improving training speed and stability while acting as a regularizer.


    self.dropout = nn.Dropout(0.2)
    self.conv1_bn = nn.BatchNorm2d(6)

<div align="center">
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/1ac0f77e-233c-47ab-a06d-5e3673cf2295" width="500"/>
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/bd8c2f66-d60b-4bb0-9a37-2a3dbd039ce6" width="500"/>
</div>

The LeNet model with regularization achieved the following performance metrics at Epoch 15:

Train Loss: 0.0113, Train Acc: 99.64%
Test Loss: 0.0273, Test Acc: 99.17%

The use of both techniques results in similar training trends compared to the base LeNet model. 

However, during the testing phase, the performance of the LeNet-Regularization model is notably improved. 

This improvement indicates that the regularization techniques effectively mitigate overfitting, leading to better generalization and higher accuracy on unseen data.

-----

## LeNet-Data Augmentation

To augment the dataset and improve model generalization, the HorizontalFlip method was utilized. 

    transforms.RandomHorizontalFlip()

This technique randomly flips images horizontally during training, effectively increasing the diversity of the training dataset.

<div align="center">
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/93fe0e62-74fe-47f7-91c8-e187bdf58592" width="500"/>
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/1cc55a04-68e1-435e-a8a3-cfd07812aac8" width="500"/>
</div>

The LeNet model with data augmentation achieved the following performance metrics at Epoch 15:

Train Loss: 0.0063, Train Acc: 99.78%
Test Loss: 0.0443, Test Acc: 98.95%

Despite the similarity in final test performance compared to LeNet-Basic, the training process with data augmentation exhibited more stable learning patterns. 

The model demonstrated enhanced robustness and was able to converge more consistently during training, indicating the effectiveness of data augmentation in promoting better generalization and reducing overfitting tendencies.


-----

## Conclusion
The experimental results demonstrate that the LeNet-Regularization model outperformed the other approaches. 

The application of regularization techniques improved the model's generalization ability. 

Furthermore, the model with data augmentation showed improved performance compared to the basic model, confirming the effectiveness of data augmentation techniques in enhancing model performance.

Feel free to explore the code and experiment with different configurations to further improve the results.
