import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from toy_dataset_and_dataloaders import train_dataset, test_dataset
from neural_networks import NeuralNetwork
from training_loop import compute_accuracy
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def prepare_dataset():
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(train_dataset)
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader

def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size) # rank is basically device id, world size is total gpus 
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank) # move model to the current gpu
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model = DDP(model, device_ids=[rank]) # wrap the model with DDP
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch) # set the epoch for the sampler, crucial for shuffling
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss.item():.2f}")

    model.eval()
    train_acc = compute_accuracy(model, train_loader, device=rank)
    print(f"[GPU{rank}] Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=rank)
    print(f"[GPU{rank}] Test accuracy", test_acc)
    destroy_process_group() # if no process group is set as parameter, the World process group. or global process group is deleted. 

if __name__ == "__main__":
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123)
    num_epochs = 3
    world_size = torch.cuda.device_count()
    if world_size > 0:
        mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    else:
        print("No CUDA GPUs detected for DDP. DDP requires CUDA GPUs.")