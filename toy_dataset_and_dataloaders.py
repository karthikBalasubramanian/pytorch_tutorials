import torch
from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

train_dataset = ToyDataset(X_train, y_train)
test_dataset = ToyDataset(X_test, y_test)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True 
)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=0
)

if __name__ == "__main__":
    print(f"second element of train_dataset: {train_dataset[1]}")
    print(f"first 2 elements of train_dataset: {train_dataset[0:2]}")
    print(f"length of train_dataset: {len(train_dataset)}")
    print(f"length of test_dataset: {len(test_dataset)}")

    print(f"one iteration of train data loader is: {next(iter(train_loader))}")
    print(f"one iteration of test data loader is: {next(iter(test_loader))}")

    for i, (x_batch, y_batch) in enumerate(train_loader):
        print(f"Batch {i}")
        print(f"Shape of x_batch: {x_batch.shape}")
        print(f"Shape of y_batch: {y_batch.shape}")