# PyTorch Starter & Practice Repository

A hands-on repository demonstrating core PyTorch concepts, custom neural network architectures, custom datasets & dataloaders, end-to-end training loops, model persistence, and CPU vs. Apple Silicon MPS GPU benchmarking.

---

## 📁 Repository Structure & Key Components

### 1. [`neural_networks.py`](./neural_networks.py)
Defines custom PyTorch neural network modules subclassing `torch.nn.Module`:
- **`NeuralNetwork`**: A 3-layer MLP (`Linear` $\rightarrow$ `ReLU` $\rightarrow$ `Linear` $\rightarrow$ `ReLU` $\rightarrow$ `Linear`) with configurable input and output feature dimensions.
- **`DeepNeuralNetwork`**: A 4-layer deep network (~10M parameters: `1024` $\rightarrow$ `2048` $\rightarrow$ `2048` $\rightarrow$ `1024` $\rightarrow$ `10`) used for high-compute performance benchmarking.
- **Key Concepts Covered**:
  - `torch.nn.Sequential` container.
  - Parameter inspection (`named_parameters()`, `requires_grad`, `.numel()`).
  - Logits vs. Probabilities (`torch.softmax(logits, dim=1)`).

---

### 2. [`toy_dataset_and_dataloaders.py`](./toy_dataset_and_dataloaders.py)
Implements PyTorch data ingestion pipeline components:
- **`ToyDataset`**: Custom dataset class inheriting from `torch.utils.data.Dataset` implementing `__getitem__` and `__len__`.
- **`DataLoader` Configuration**:
  - **Batching (`batch_size`)**: Grouping tensor samples for vector processing.
  - **Shuffling (`shuffle=True`)**: Randomizing sample order per epoch to break sequence bias and satisfy I.I.D. assumptions.
  - **Drop Last (`drop_last`)**: Handling trailing incomplete mini-batches.
  - **Multi-processing (`num_workers`)**: Parallelizing data loading background processes.

---

### 3. [`training_loop.py`](./training_loop.py)
A complete, end-to-end PyTorch model training and evaluation script:
- **Optimizer (`torch.optim.SGD`)**: Stochastic Gradient Descent weight and bias parameter updates ($w \leftarrow w - \eta \cdot \nabla w$).
- **Loss Function (`torch.nn.functional.cross_entropy`)**: Multi-class cross-entropy loss calculation on raw model logits.
- **Backpropagation Steps**:
  1. `optimizer.zero_grad()`: Reset accumulated gradients before computing fresh slopes.
  2. `loss.backward()`: Compute gradients via automatic differentiation (`autograd`).
  3. `optimizer.step()`: Update network parameters.
- **Model Evaluation (`compute_accuracy`)**: Running in evaluation mode (`model.eval()`, `torch.no_grad()`).
- **Persistence (`torch.save` & `torch.load`)**: Saving model weights (`state_dict`) to `.pth` file and restoring them.

---

### 4. [`benchmark_devices.py`](./benchmark_devices.py)
A performance benchmarking suite comparing **CPU vs. Apple Silicon MPS GPU (`torch.backends.mps`)**:
- Proper GPU timing measurement using **`torch.mps.synchronize()`**.
- Benchmark analysis showing:
  - **Small networks / small datasets**: CPU is faster (~5x) due to zero kernel dispatch overhead.
  - **Large deep networks / heavy matrix math**: MPS GPU is significantly faster (~1.6x to 6.8x) due to massive parallel matrix throughput.

---

### 5. [`distributed_data_processing.py`](./distributed_data_processing.py)
A complete **Distributed Data Parallel (DDP)** multi-GPU training implementation:
- **Process Spawning (`mp.spawn`)**: Launches $N$ parallel worker processes assigning unique GPU ranks (`rank=0, 1...`).
- **Rendezvous & TCP Handshake (`init_process_group`)**:
  - `MASTER_ADDR` & `MASTER_PORT` environment variables configure the coordination endpoint.
  - `init_process_group()` executes a **blocking C++ handshake** via `c10d::TCPStore`. Master (Rank 0) collects IP addresses from all workers, compiles a global directory, and initializes the NCCL P2P ring communicator.
- **Data Sharding (`DistributedSampler`)**: Strided index partitioning (`indices[rank::world_size]`) to assign non-overlapping dataset shards per GPU.
- **Epoch Reshuffling (`sampler.set_epoch`)**: Deterministically seeds random index shuffling across all GPUs per epoch.
- **Gradient AllReduce (`torch.nn.parallel.DistributedDataParallel`)**: Automatically averages gradients across GPUs over NCCL peer-to-peer ring during `loss.backward()` before `optimizer.step()`.

---

## 🚀 Execution Commands

```bash
# Inspect neural network architectures and parameters
python neural_networks.py

# Run dataset and dataloader pipeline
python toy_dataset_and_dataloaders.py

# Run full training loop and save/load model
python training_loop.py

# Run CPU vs. MPS GPU timing benchmarks
python benchmark_devices.py

# Run Distributed Data Parallel multi-GPU script (requires CUDA GPUs)
python distributed_data_processing.py
```
