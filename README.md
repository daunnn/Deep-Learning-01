# Deep-Learning_HW2

This repository contains the implementation of the LeNet model using different approaches and compares their results. 

Table of Contents
+ LeNet-Basic
+ CustomMLP
+ LeNet-Regularization
+ LeNet-Data Augmentation
+ Conclusion

-------

LeNet-Basic
<div align="center">
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/bd14d1a6-28ee-4596-89b0-2fa7db7dc8c2" width="500"/>
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/50d9ea03-0795-478d-9220-042acb90677b" width="500"/>
</div>


The basic LeNet model achieved the following performance metrics at Epoch 15:

Train Loss: 0.0069, Train Acc: 99.77%
Test Loss: 0.0540, Test Acc: 98.91%

![image](https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/3056a6ff-c1a3-4b67-aed5-ff28accab581)
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

CustomMLP
<div align="center">
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/b6274945-3874-4ad9-8fbd-93c9e23a0967" width="500"/>
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/b39a8880-9497-4340-ad63-66801a1fc84e" width="500"/>
</div>

The CustomMLP model achieved the following performance metrics at Epoch 15:

Train Loss: 0.0141, Train Acc: 99.53%
Test Loss: 0.1059, Test Acc: 97.57%

------

LeNet-Regularization
<div align="center">
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/1ac0f77e-233c-47ab-a06d-5e3673cf2295" width="500"/>
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/bd8c2f66-d60b-4bb0-9a37-2a3dbd039ce6" width="500"/>
</div>

The LeNet model with regularization achieved the following performance metrics at Epoch 15:

Train Loss: 0.0113, Train Acc: 99.64%
Test Loss: 0.0273, Test Acc: 99.17%

-----

LeNet-Data Augmentation
<div align="center">
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/93fe0e62-74fe-47f7-91c8-e187bdf58592" width="500"/>
  <img src="https://github.com/daunnn/Deep-Learning_HW2/assets/98380084/1cc55a04-68e1-435e-a8a3-cfd07812aac8" width="500"/>
</div>

The LeNet model with data augmentation achieved the following performance metrics at Epoch 15:

Train Loss: 0.0063, Train Acc: 99.78%
Test Loss: 0.0443, Test Acc: 98.95%

-----

Conclusion
The experimental results demonstrate that the LeNet-Regularization model outperformed the other approaches. The application of regularization techniques improved the model's generalization ability. Furthermore, the model with data augmentation showed improved performance compared to the basic model, confirming the effectiveness of data augmentation techniques in enhancing model performance.

Feel free to explore the code and experiment with different configurations to further improve the results.
