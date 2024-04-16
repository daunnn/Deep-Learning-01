import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(), # 이미지의 픽셀 값을 [0,1]범위의 텐서 형태로 변환
            transforms.Normalize(mean=[0.1307], std=[0.3081]) #평균을 빼고 표준편차로 나누어주는 정규화 작업 진행 
        ])
        self.image_files = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path)
        image = self.transform(image)
        label = int(img_name.split('_')[1].split('.')[0])
        return image, label

if __name__ == '__main__':
    # Test codes to verify the implementations
    data_dir = "../data/train/" #현재는 tmp data dir로 train 지정해둠 -> main.py에서 호출할 때 새롭게 정의 가능 
    mnist_dataset = MNIST(data_dir)

    # Test __len__
    print(len(mnist_dataset)) # print 결과 : 60000


    # Test __getitem__
    img, label = mnist_dataset[0]
    print(img.shape, label) #print 결과 : torch.Size([1, 28, 28]) 6


