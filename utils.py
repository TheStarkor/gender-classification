import PIL
import torchvision

import matplotlib.pyplot as plt

def get_image(path):
    return PIL.Image.open(path)

def image_to_tensor(image):
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])

    return tf(image)

def show_data(data):
    plt.style.use('dark_background')
    plt.figure(figsize=(16, 4))
    images, labels = next(iter(data))
    label_to_title = {0: 'female(0)', 1: 'male(1)'}
    for index in range(16): # batch size
        plt.subplot(2, 8, index + 1)
        plt.title(label_to_title[labels[index].numpy().item()])
        plt.imshow(images[index].permute(1,2,0).numpy())
        plt.axis("off")
    plt.show()
