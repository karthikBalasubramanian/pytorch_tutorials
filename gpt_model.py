import torch
from torch import nn
import tiktoken
from attention_mechanism import MultiHeadAttention

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Size of our vocabulary (tokens)
    "context_length": 1024, # Maximum sequence length (number of input tokens)
    "emb_dim": 768,         # Embedding dimension size (hidden state dimension)
    "n_heads": 12,          # Number of attention heads in multi-head attention
    "n_layers": 12,         # Number of transformer block layers
    "drop_rate": 0.1,       # Dropout rate for regularization (10% probability)
    "qkv_bias": False,      # Query-Key-Value bias option (whether linear projections use bias)
}

class LayerNorm(nn.Module):
    """
    Layer Normalization module.
    
    What it does:
        Normalizes activations across the feature dimension (emb_dim) for each 
        token in a sequence to have mean = 0.0 and variance = 1.0.
        
    Why it is used:
        - In a deep model with 12 or 24 Transformer layers, outputs from layer 1 
          become inputs to layer 2, which become inputs to layer 3, and so on.
          LayerNorm prevents activations from exploding or shrinking across these deep layers.
        - Smooths the loss landscape, enabling higher learning rates and faster convergence.
        - Decouples layer scale so each layer receives consistently normalized inputs.
        - Includes learnable scale (gain) and shift (bias) parameters so the model can 
          adjust feature magnitude and offset if strict mean=0 / var=1 normalization needs adaptation.

        
    Learnable Parameters (scale & shift):
        - self.scale (gain): Initialized to 1.0 (ones). Backpropagation updates each feature's
          scale factor (can become > 1.0 or < 1.0).
        - self.shift (bias): Initialized to 0.0 (zeros). Backpropagation updates each feature's
          shift value (can become > 0.0 or < 0.0).
        - Purpose: Allows the model to learn the optimal feature scale (gain) and shift (bias) if 
          strict mean=0 and var=1 normalization needs adaptation for model performance.
        - Applied via element-wise operations: (self.scale * norm_x + self.shift).
        
    Where it is used in GPT architecture:
        - 2 times per Transformer block (before Attention & before MLP).
        - 1 final time (final_norm_layer) before the output linear layer.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim)) # learnable gain parameter for each feature
        self.shift = nn.Parameter(torch.zeros(embed_dim)) # learnable bias parameter for each feature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # skip bessel's correction
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) / Multi-Layer Perceptron (MLP) module.

    What it does:
        Processes each token's hidden representation vector (the activation output from 
        the Multi-Head Attention sub-layer) independently across sequence positions.


    Architecture & Dimension Transformation:
        1. Expansion Layer (Linear):
           - Expands feature dimension by 4x (emb_dim -> 4 * emb_dim, e.g., 768 -> 3072).
           - Purpose: Increases model capacity to act as a key-value memory bank for factual knowledge.
           - Projects data into higher-dimensional space where non-linear patterns are easier to separate (Cover's Theorem).
           
        2. Non-Linear Activation (GELU):
           - Choice of GELU (Gaussian Error Linear Unit): A smooth, differentiable alternative to ReLU.
           - Behavior: Positive values are kept intact (x -> x), small negative values (-1 to 0) 
             dip smoothly down to ~ -0.17, and large negative values approach zero (x < -2.0 -> 0).
           - Avoids hard zeros and dead neurons, enabling smoother gradients and stable convergence.
           - Uses Tanh approximation (approximate="tanh") for computational efficiency (GPT-2/3 standard).

           
        3. Compression Layer (Linear):
           - Contracts feature dimension back by 4x (4 * emb_dim -> emb_dim, e.g., 3072 -> 768).
           - Purpose: Summarizes high-dimensional features back to emb_dim so tensor shape 
             matches for residual additions (x = x + ffn(x)).
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
        )
    
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            num_heads=config[  "n_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"]
        )
        self.feed_forward = FeedForward(config)
        self.norm1 = LayerNorm(embed_dim=config["emb_dim"])
        self.norm2 = LayerNorm(embed_dim=config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = shortcut + x

        return x

class GPTModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # Layers / Modules
        self.tok_emb_layer = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb_layer = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_layer = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["n_layers"])])
        self.final_norm_layer = LayerNorm(config["emb_dim"])
        self.out_head_layer = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)



    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Retrieve embedding tensors
        tok_embeds = self.tok_emb_layer(input_ids)
        pos_embeds = self.pos_emb_layer(torch.arange(seq_len, device=input_ids.device))
        x = tok_embeds + pos_embeds
        x = self.drop_layer(x)
        x = self.transformer_blocks(x)

        # 1. Capture stats BEFORE LayerNorm
        mean_before = x.mean(dim=-1, keepdim=True)
        var_before = x.var(dim=-1, keepdim=True, unbiased=False)

        # 2. Apply LayerNorm
        x = self.final_norm_layer(x)

        # 3. Capture stats AFTER LayerNorm
        mean_after = x.mean(dim=-1, keepdim=True)
        var_after = x.var(dim=-1, keepdim=True, unbiased=False)

        # 4. Clean Comparison Print
        torch.set_printoptions(sci_mode=False, precision=4)
        print("\n--- LayerNorm Verification ---")
        print(f"Mean BEFORE LayerNorm (token 0): {mean_before[0, 0].item():.4f}")
        print(f"Var  BEFORE LayerNorm (token 0): {var_before[0, 0].item():.4f}")
        print(f"Mean AFTER  LayerNorm (token 0): {mean_after[0, 0].item():.4f}")
        print(f"Var  AFTER  LayerNorm (token 0): {var_after[0, 0].item():.4f}\n")
        torch.set_printoptions(sci_mode=None)

        logits = self.out_head_layer(x)
        return logits






if __name__ == "__main__":

    tokenizer = tiktoken.get_encoding("gpt2") # download the tokenizer
    batch = ["Every effort moves you", "Every day holds a"] # both encode to 4 tokens
    input_list = [tokenizer.encode(s) for s in batch] # returns list of python lists
    inputs = torch.tensor(input_list) # convert list of lists directly to a PyTorch tensor

    print("Input:", inputs)
    print("Input shape:", inputs.shape)

    torch.manual_seed(123) # set the seed for random number generation.
    model_config = GPT_CONFIG_124M
    model = GPTModel(model_config) # create the model
    logits = model(inputs) # forward pass
    print("Logits shape:", logits.shape)
    print(logits)

    # -------------------------------------------------------------------------
    # 1. Total parameters in GPT-2 (Untied vs. Weight Tying)
    # -------------------------------------------------------------------------
    # Weight Tying (Press & Wolf 2017 / GPT-2 standard):
    # Links out_head_layer.weight directly to tok_emb_layer.weight so both layers
    # share the exact same 38.6M weight matrix in memory.
    #
    # Intuition:
    #   - tok_emb_layer maps Token ID -> 768-dim Vector (Input lookup).
    #   - out_head_layer computes dot products (h * v_token) between final hidden 
    #     state 'h' and token vector 'v_token' to score next-word probability.
    #   - Using the exact same representation matrix in reverse is symmetric, 
    #     saves 38.6M parameters (163M -> 124M), and prevents output overfitting.
    total_params_untied = sum(p.numel() for p in model.parameters())
    
    # Tie weights between out_head_layer and tok_emb_layer
    model.out_head_layer.weight = model.tok_emb_layer.weight
    total_params_tied = sum(p.numel() for p in model.parameters())


    print("\n" + "=" * 60)
    print("1. TOTAL PARAMETER COUNT & WEIGHT TYING")
    print("=" * 60)
    print(f"   Total Parameters (Untied)            : {total_params_untied:,}")
    print(f"   Total Parameters (With Weight Tying) : {total_params_tied:,}")
    print(f"   Parameters Saved via Weight Tying    : {total_params_untied - total_params_tied:,}")

    # -------------------------------------------------------------------------
    # 2. Parameter comparison: FeedForward (MLP) vs MultiHeadAttention
    # -------------------------------------------------------------------------
    attn_params_per_block = sum(p.numel() for p in model.transformer_blocks[0].attention.parameters())
    ffn_params_per_block = sum(p.numel() for p in model.transformer_blocks[0].feed_forward.parameters())
    total_attn_params = attn_params_per_block * model_config["n_layers"]
    total_ffn_params = ffn_params_per_block * model_config["n_layers"]

    print("\n" + "=" * 60)
    print("2. MODULE PARAMETER COMPARISON (Attention vs FeedForward)")
    print("=" * 60)
    print(f"   Per Block  - MultiHeadAttention      : {attn_params_per_block:,}")
    print(f"   Per Block  - FeedForward (MLP)       : {ffn_params_per_block:,}")
    print(f"   All 12 Blks- MultiHeadAttention      : {total_attn_params:,}")
    print(f"   All 12 Blks- FeedForward (MLP)       : {total_ffn_params:,}")
    print(f"   Ratio (FFN / Attention)              : {ffn_params_per_block / attn_params_per_block:.2f}x")

    # -------------------------------------------------------------------------
    # 3. Compute Memory Requirements for Parameters (float32 = 4 bytes per param)
    # -------------------------------------------------------------------------
    bytes_per_param = 4  # 32-bit floating point precision
    total_bytes_untied = total_params_untied * bytes_per_param
    total_bytes_tied = total_params_tied * bytes_per_param

    print("\n" + "=" * 60)
    print("3. PARAMETER MEMORY FOOTPRINT (float32)")
    print("=" * 60)
    print(f"   Untied Model (163M params)           : {total_bytes_untied / (1024**2):.2f} MB ({total_bytes_untied / (1024**3):.4f} GB)")
    print(f"   Tied Model   (124M params)           : {total_bytes_tied / (1024**2):.2f} MB ({total_bytes_tied / (1024**3):.4f} GB)")
    print("=" * 60 + "\n")



