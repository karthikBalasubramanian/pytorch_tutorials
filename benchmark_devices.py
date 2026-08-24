import time
import torch
import torch.nn.functional as F

from neural_networks import NeuralNetwork, DeepNeuralNetwork
from toy_dataset_and_dataloaders import train_loader, ToyDataset, DataLoader

def train_model(device_name, num_epochs=500, loader=train_loader):
    device = torch.device(device_name)
    torch.manual_seed(234)
    model = NeuralNetwork(num_inputs=2, num_outputs=2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Warmup step to ensure GPU/kernels are initialized
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        loss = F.cross_entropy(model(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device_name == "mps":
        torch.mps.synchronize()

    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Synchronize GPU to ensure all operations finish before stopping the timer
    if device_name == "mps":
        torch.mps.synchronize()

    elapsed_time = time.perf_counter() - start_time
    return elapsed_time


if __name__ == "__main__":
    print("=" * 55)
    print("      PYTORCH CPU vs MPS (Mac GPU) TIMING BENCHMARK")
    print("=" * 55)

    # Benchmark 1: Small dataset (Toy dataset: 4 samples, batch_size=2)
    epochs_small = 1000
    print(f"\n--- Benchmark 1: Small Dataset (4 samples, 1000 epochs) ---")
    
    cpu_time_small = train_model("cpu", num_epochs=epochs_small)
    print(f"CPU Time: {cpu_time_small:.4f} seconds")

    if torch.backends.mps.is_available():
        mps_time_small = train_model("mps", num_epochs=epochs_small)
        print(f"MPS Time: {mps_time_small:.4f} seconds")
        ratio = mps_time_small / cpu_time_small
        if ratio > 1.0:
            print(f"--> CPU was {ratio:.2f}x FASTER than MPS for this tiny dataset!")
        else:
            print(f"--> MPS was {1/ratio:.2f}x FASTER than CPU!")
    else:
        print("MPS is not available on this system.")

    # Benchmark 2: Larger Synthetic Dataset (10,000 samples, batch_size=64)
    print(f"\n--- Benchmark 2: Larger Dataset (10,000 samples, 20 epochs) ---")
    X_large = torch.randn(10000, 2)
    y_large = torch.randint(0, 2, (10000,))
    large_dataset = ToyDataset(X_large, y_large)
    large_loader = DataLoader(large_dataset, batch_size=64, shuffle=True)

    cpu_time_large = train_model("cpu", num_epochs=20, loader=large_loader)
    print(f"CPU Time: {cpu_time_large:.4f} seconds")

    if torch.backends.mps.is_available():
        mps_time_large = train_model("mps", num_epochs=20, loader=large_loader)
        print(f"MPS Time: {mps_time_large:.4f} seconds")
        ratio_large = cpu_time_large / mps_time_large
        if ratio_large > 1.0:
            print(f"--> MPS was {ratio_large:.2f}x FASTER than CPU for this dataset!")
        else:
            print(f"--> CPU was {1/ratio_large:.2f}x FASTER than MPS!")

    # Benchmark 3: Heavy Computation (Large Matrices: 512x4096 @ 4096x4096)
    print(f"\n--- Benchmark 3: Heavy Computation (512x4096 @ 4096x4096 MatMul x 100 iterations) ---")
    x_cpu = torch.randn(512, 4096)
    w_cpu = torch.randn(4096, 4096)

    # CPU
    t0 = time.perf_counter()
    for _ in range(100):
        _ = torch.matmul(x_cpu, w_cpu)
    t_cpu_heavy = time.perf_counter() - t0
    print(f"CPU Time: {t_cpu_heavy:.4f} seconds")

    # MPS
    if torch.backends.mps.is_available():
        x_mps = x_cpu.to("mps")
        w_mps = w_cpu.to("mps")
        _ = torch.matmul(x_mps, w_mps) # warmup
        torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(100):
            _ = torch.matmul(x_mps, w_mps)
        torch.mps.synchronize()
        t_mps_heavy = time.perf_counter() - t0
        print(f"MPS Time: {t_mps_heavy:.4f} seconds")
        print(f"--> MPS was {t_cpu_heavy / t_mps_heavy:.2f}x FASTER than CPU!")

    # Benchmark 4: Deep Neural Network Training (Full forward, backward & SGD steps)
    print(f"\n--- Benchmark 4: Deep Neural Network Training (1024 inputs, 2048 hidden units, 10M params) ---")
    def run_deep_training(device_name, num_steps=50):
        device = torch.device(device_name)
        torch.manual_seed(123)
        model = DeepNeuralNetwork().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        X = torch.randn(256, 1024)
        y = torch.randint(0, 10, (256,))

        # Warmup step
        X_d, y_d = X.to(device), y.to(device)
        loss = F.cross_entropy(model(X_d), y_d)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device_name == 'mps':
            torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(num_steps):
            model.train()
            X_d, y_d = X.to(device), y.to(device)
            logits = model(X_d)
            loss = F.cross_entropy(logits, y_d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if device_name == 'mps':
            torch.mps.synchronize()

        return time.perf_counter() - t0

    t_cpu_deep = run_deep_training('cpu')
    print(f"CPU Time: {t_cpu_deep:.4f} seconds")

    if torch.backends.mps.is_available():
        t_mps_deep = run_deep_training('mps')
        print(f"MPS Time: {t_mps_deep:.4f} seconds")
        print(f"--> MPS was {t_cpu_deep / t_mps_deep:.2f}x FASTER than CPU for Deep NN Training!")

    print("\n" + "=" * 55)


