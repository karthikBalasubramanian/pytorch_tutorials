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

### 6. [`simple_tokenizer.py`](./simple_tokenizer.py)
A word and punctuation tokenizer (`SimpleTokenizerV1`) built for Large Language Models (LLMs):
- **Token Extraction**: Uses regular expressions `re.split(r'([,.?_!"()\']|--|\s)', text)` to split text into words and individual punctuation tokens.
- **Vocabulary Construction**: Builds a token-level `vocab` dictionary (`token -> integer ID`) from preprocessed text (`the-verdict.txt`).
- **Encoding & Decoding**:
  - `encode(text)`: Converts string tokens to integer IDs (`self.str_to_int[token]`).
  - `decode(ids)`: Converts integer IDs back into readable string text (`self.int_to_str[id]`).

---

### 7. [`gpt_dataset_and_embeddings.py`](./gpt_dataset_and_embeddings.py)
Implements an end-to-end dataset pipeline, token embedding, un-embedding/decoding, and positional embedding injection for GPT architectures:
- **Autoregressive Dataset & DataLoader (`GPTDatasetV1`, `create_data_loader_v1`)**:
  - Tokenizes raw text using BPE (`tiktoken` GPT-2 tokenizer).
  - Uses sliding windows (`max_length` context chunks, `stride`) to construct shifted input-target training pairs `(input_ids, target_ids)` of shape `[batch_size, sequence_length]`.
- **3D Token Embedding Lookup (`nn.Embedding`)**:
  - Converts 2D token IDs `[batch_size, sequence_length]` into 3D feature tensors `[batch_size, sequence_length, embedding_dim]` (e.g. `[8, 4, 768]`).
- **Un-embedding & Logit Projection**:
  - Demonstrates un-embedding via matrix multiplication against the embedding table (`embedded_inputs @ weight.T`) to compute vocabulary logits `[batch_size, sequence_length, vocab_size]`.
  - Extracts token IDs via `torch.argmax(dim=-1)` and decodes them back to string text sequences.
- **Positional Embedding & Broadcasting Addition**:
  - Explains why self-attention requires positional encoding (permutation invariance).
  - Creates a positional embedding lookup table (`context_length` capacity = 256) and adds position vectors `[sequence_length, embedding_dim]` to token embeddings via PyTorch broadcasting to produce combined `input_embeddings` containing both semantic identity and word order.

---

### 8. [`attention_mechanism.py`](./attention_mechanism.py)
Implements **Part 1: Simplified Dot-Product Attention** (without weight parameters/gradients) along with an in-depth conceptual writeup:

#### 💡 Core Theoretical Concepts

1. **What is Attention?**
   - **Static vs. Contextual Embeddings**: Standard lookup tables (`nn.Embedding`) assign a single fixed vector representation to a token regardless of context (e.g., "bank" in "river bank" vs. "bank account"). Attention allows tokens to dynamically interact, query surrounding tokens, and aggregate contextual information into dynamic **context vectors**.
   - **Mechanism**: Every token looks at all other tokens in a sequence, calculates relevance weights (how much to pay attention to each token), and computes a weighted average of token embeddings to update its own representation.

2. **Why Dot Product?**
   - **Geometric Similarity**: The dot product between two vectors $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos \theta$ measures their directional alignment in high-dimensional embedding space.
     - Vectors pointing in similar directions yield high positive dot products.
     - Orthogonal vectors yield zero dot product.
     - Opposing vectors yield negative dot products.
   - **Matrix Parallelism**: Pairwise dot products for an entire sequence $X \in \mathbb{R}^{T \times d}$ can be calculated simultaneously as a single matrix multiplication $A_{raw} = X X^T \in \mathbb{R}^{T \times T}$, leveraging fast GPU matrix multiplication (GEMM/BLAS).

3. **What is Normalization?**
   - Raw dot product scores are unconstrained real numbers $(-\infty, \infty)$ scaling with vector magnitude and dimension.
   - Normalization converts raw scores into valid **probability distributions** (weights $\in [0, 1]$ summing to $1.0$ across each row).
   - This ensures the resulting context vector $\mathbf{z}_i = \sum_{j} w_{ij} \mathbf{x}_j$ is a **convex combination** (weighted average) of input vectors, preserving the numerical scale of input features without numerical explosion.

4. **Why Softmax Normalization vs. Naive Summation ($w_i = \frac{a_i}{\sum a}$)?**
   - **Handling Negative Dot Products**: Dot products can be negative. Naive summation ($\frac{a_{ij}}{\sum_k a_{ik}}$) breaks severely:
     - If the denominator sum is negative, weight signs invert.
     - If the denominator sum is zero, division by zero occurs ($NaN$).
     - Individual naive weights can become negative or greater than $1.0$.
   - **Guaranteed Valid Probability Distribution**: Exponentiation ($\exp(x)$) maps all real numbers $(-\infty, \infty)$ strictly into positive real values $(0, \infty)$. Softmax guarantees $w_{ij} \in (0, 1)$ and $\sum_{j} w_{ij} = 1.0$.
   - **Non-linear Sharpness / Focus**: $\exp(x)$ exponentially magnifies large dot-product differences, sharpening focus on highly relevant tokens while suppressing noise from less relevant ones.
   - **Smooth Differentiability**: Softmax provides smooth, continuous gradients everywhere ($\frac{\partial s_i}{\partial a_j} = s_i (\delta_{ij} - s_j)$), ideal for backpropagation.
   - **Numerical Stability**: Standard implementations compute $\exp(x_i - \max(x))$ to prevent exponential overflow.

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

# Run simple LLM word & punctuation tokenizer
python simple_tokenizer.py

# Run GPT dataset loader, token embeddings, decoding & positional embeddings
python gpt_dataset_and_embeddings.py

# Run simplified dot-product self-attention without weights
python attention_mechanism.py
```


