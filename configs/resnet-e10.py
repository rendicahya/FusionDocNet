from models.resnet import ResNet
from torchvision import transforms

num_classes = 15
model = ResNet(num_classes=num_classes)
epochs = 10
data_dir = "data"
batch_size = 16
num_workers = 8
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
