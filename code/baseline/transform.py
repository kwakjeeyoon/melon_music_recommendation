
   
from torchvision import transforms

def get_transform():

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform