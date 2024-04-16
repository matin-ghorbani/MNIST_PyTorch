import torch
from torchvision.transforms import ToTensor

from PIL import Image

from train import ImageClassifier

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = ImageClassifier().to(device)

    clf.load_state_dict(torch.load('./mnist_classifier_torch.pth', map_location=torch.device(device)))

    img = Image.open('./io/input/img_1.jpg')

    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(clf(img_tensor)))
