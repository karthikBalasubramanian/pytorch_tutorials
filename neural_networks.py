import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )
    

    def forward(self, x):
        logits = self.layers(x)
        return logits


class DeepNeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs=1024, num_outputs=10):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits



if __name__ == "__main__":
    model = NeuralNetwork(50, 3)
    print(model)
    
    # Print parameter details
    for name, param in model.named_parameters():
        print(f"{name:16s} | shape: {str(list(param.shape)):10s} | numel: {param.numel():4d} | requires_grad: {param.requires_grad}")

    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Total number of trainable model parameters:", num_params)
    print(model.layers[0].weight)

    torch.manual_seed(123)
    X= torch.rand(1, 50)
    output = model(X)
    print(output)
    print(output.shape)

    with torch.no_grad():
        output_nograd = model(X)
        print(output_nograd)

        out_no_grad_class = torch.softmax(output_nograd, dim=1)
        print(out_no_grad_class)
    


