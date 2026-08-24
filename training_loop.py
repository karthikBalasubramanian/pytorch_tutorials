import torch
import torch.nn.functional as F

from neural_networks import NeuralNetwork
from toy_dataset_and_dataloaders import train_loader, test_loader

def compute_accuracy(model, dataloader, device=torch.device("cpu")):

    model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


if __name__ == "__main__":
    torch.manual_seed(234)

    model = NeuralNetwork(num_inputs=2, num_outputs=2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            logits = model(features) # make predictions

            loss = F.cross_entropy(logits, labels) # calculate how wrong the model is

            optimizer.zero_grad() # clear out old gradients
            loss.backward() # compute gradients
            optimizer.step() # update weights

            ### LOGGING
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx+1:03d}/{len(train_loader):03d}"
            f" | Train Loss {loss.item():.4f}")

    print("\n" + "=" * 30)
    print(f"Train accuracy: {compute_accuracy(model, train_loader, device)}")
    print(f"Test accuracy: {compute_accuracy(model, test_loader, device)}")
    print("=" * 30 + "\n")

    save_model(model, "nn_model.pth")

    # Load model
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.load_state_dict(torch.load("nn_model.pth"))
    model.eval()

    print(f"Test accuracy after loading model: {compute_accuracy(model, test_loader)}")