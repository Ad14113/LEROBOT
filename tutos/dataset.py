
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset



delta_timestamps = {
    "observation.images.side": [-0.2, -0.1, 0.0], 
    "observation.images.up": [-0.2, -0.1, 0.0],
}


dataset = LeRobotDataset (
    "lerobot/svla_so101_pickplace",
    delta_timestamps = delta_timestamps,
    video_backend="pyav"
)

print("_____________DATASET_____________")
print(dataset)
print("_________________________________")

batch_size = 16
dataloader = DataLoader(
    dataset, 
    batch_size = batch_size
)

num_epochs = 1
for epoch in range(num_epochs):
    for batch in dataloader:
        print("_____________BATCH_____________")
        #print(batch["action"])
        print("_________________________________")



