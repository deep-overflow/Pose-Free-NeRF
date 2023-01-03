import yaml
from utils import Config
from dataset import CO3D
from torch.utils.data import DataLoader
from torchvision import transforms

with open('configs.yaml') as f:
    configs_dict = yaml.load(f, Loader=yaml.FullLoader)

configs = Config(configs_dict)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((configs.dataset.transforms.img_h, configs.dataset.transforms.img_w)),
])
train_dataset = CO3D(configs, transform=transform)

# images, labels = train_dataset[0]

# print(images.shape)
# print(labels.shape)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=configs.dataset.batch_size,
    shuffle=True
)

for images, poses in train_dataloader:
    print(images.shape)
    print(poses.shape)
    break
