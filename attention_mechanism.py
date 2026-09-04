"""
Attention Mechanism (Part 1: Simplified Dot-Product Attention without Weights)

This script demonstrates basic self-attention using a simple 2D input tensor without trainable 
weight parameters (W_Q, W_K, W_V). It illustrates raw dot-product similarity computation, 
compares naive summation normalization against Softmax normalization, and computes final 
context vectors.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def simplified_attention_demo():
    print("=" * 70)
    print("PART 1: SIMPLIFIED DOT-PRODUCT ATTENTION (NO WEIGHT GRADIENTS)")
    print("=" * 70)

    # 1. Define a simple 2D tensor of inputs [seq_len, embedding_dim]
    # Representing a sequence of 3 tokens, each with a 4-dimensional embedding vector
    inputs = torch.tensor([
        [0.43, 0.15, 0.89, 0.55],  # Token 1 (e.g., "Your")
        [0.55, 0.87, 0.66, 0.23],  # Token 2 (e.g., "journey")
        [0.57, 0.85, 0.08, 0.12]   # Token 3 (e.g., "starts")
    ], dtype=torch.float32)

    seq_len, embedding_dim = inputs.shape
    print(f"\n1. Input Tensor (X): Shape [{seq_len}, {embedding_dim}]")
    print(inputs)

    # ----------------------------------------------------
    # 2. Compute Raw Similarity Scores via Dot Product (Phase 1)
    # ----------------------------------------------------
    # Phase 1: Dot products between input vectors (X @ X.T)
    # Measure directional similarity/alignment between query token x_i and key token x_j.
    # Score a_ij = x_i . x_j  (Vector-Vector Dot Product -> scalar similarity score)
    raw_scores = inputs @ inputs.T  # Shape: [seq_len, seq_len]

    print(f"\n2. Raw Attention Scores (A = X @ X^T): Shape [{seq_len}, {seq_len}]")
    print(raw_scores)
    
    # Detailed view for query token 2 (index 1)
    query_idx = 1
    query_vector = inputs[query_idx]
    print(f"\n   Example for Token {query_idx + 1} (query vector = {query_vector}):")
    for j in range(seq_len):
        score = torch.dot(query_vector, inputs[j])
        print(f"   - Dot product with Token {j + 1}: {score:.4f}")

    # ----------------------------------------------------
    # 3. Normalization: Naive Summation vs. Softmax
    # ----------------------------------------------------
    print("\n3. Normalization Comparison:")
    
    # Naive Summation Normalization (summing to 1 by simple division)
    naive_sum_weights = raw_scores / raw_scores.sum(dim=-1, keepdim=True)
    print("\n   a) Naive Summation Normalization (raw_scores / sum):")
    print(naive_sum_weights)
    print(f"      Row sums: {naive_sum_weights.sum(dim=-1)}")

    # Softmax Normalization
    softmax_weights = F.softmax(raw_scores, dim=-1)
    print("\n   b) Softmax Normalization (torch.softmax):")
    print(softmax_weights)
    print(f"      Row sums: {softmax_weights.sum(dim=-1)}")

    # Demonstration: Why Naive Summation Fails with Negative Similarity Scores
    inputs_with_negatives = torch.tensor([
        [ 1.0, -2.0,  0.5],
        [-1.0,  1.5, -0.5],
        [ 0.2,  0.1, -1.2]
    ], dtype=torch.float32)
    
    raw_scores_neg = inputs_with_negatives @ inputs_with_negatives.T
    print("\n   c) Edge Case Demo: Tensors with Negative Components")
    print("      Raw Scores Matrix:")
    print(raw_scores_neg)
    
    # Check naive sum issue:
    naive_denom = raw_scores_neg.sum(dim=-1, keepdim=True)
    print(f"      Naive Sum Denominators per row: {naive_denom.squeeze()}")
    naive_weights_neg = raw_scores_neg / naive_denom
    print("      Naive Weights (can contain negative values or division issues):")
    print(naive_weights_neg)

    softmax_weights_neg = F.softmax(raw_scores_neg, dim=-1)
    print("      Softmax Weights (guaranteed valid probability distribution in [0, 1]):")
    print(softmax_weights_neg)

    # ----------------------------------------------------
    # 4. Compute Context Vectors: Weighted Sum (Phase 2)
    # ----------------------------------------------------
    # Phase 2: Matrix multiplication of Attention Weights @ Input Vectors (W_attn @ X)
    # Context vector z_i is a weighted sum (linear combination) of all input vectors:
    # z_i = w_i1*x_1 + w_i2*x_2 + ... + w_iT*x_T  -->  Z = Softmax_Weights @ X
    # Note: While '@' performs matrix multiplication (dot products along inner dimensions),
    # conceptually this step uses the softmax probabilities to scale and blend input feature vectors!
    context_vectors = softmax_weights @ inputs  # Shape: [seq_len, embedding_dim]

    print(f"\n4. Output Context Vectors (Z = Attention_Weights @ X): Shape [{seq_len}, {embedding_dim}]")
    print(context_vectors)

    print("\nSummary of Shapes:")
    print(f"- Inputs (X):            {list(inputs.shape)}")
    print(f"- Raw Scores (X @ X^T):  {list(raw_scores.shape)}")
    print(f"- Attention Weights (W): {list(softmax_weights.shape)}")
    print(f"- Context Vectors (Z):   {list(context_vectors.shape)}")
    print("=" * 70)



class SelfAttention_v1(nn.Module):
    """
    Self-Attention V1: Uses manual nn.Parameter weight matrices W_Q, W_K, W_V.
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_Q = nn.Parameter(torch.rand(self.d_in, self.d_out))
        self.W_K = nn.Parameter(torch.rand(self.d_in, self.d_out))
        self.W_V = nn.Parameter(torch.rand(self.d_in, self.d_out))
        
    def forward(self, x):
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V
        attn_scores = Q @ K.T
        attn_weights = F.softmax(attn_scores / math.sqrt(self.d_out), dim=-1)
        context_vector = attn_weights @ V
        return context_vector


class SelfAttention_v2(nn.Module):
    """
    Self-Attention V2: Uses PyTorch's nn.Linear for automatic weight initialization,
    optional bias support, and transpose(-2, -1) for 2D & 3D batch safety.
    """
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # nn.Linear handles proper weight initialization (Kaiming / Xavier uniform)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)  # Shape: [..., seq_len, d_out]
        keys    = self.W_key(x)    # Shape: [..., seq_len, d_out]
        values  = self.W_value(x)  # Shape: [..., seq_len, d_out]

        # transpose(-2, -1) swaps the last two dimensions safely for 2D and 3D batched inputs
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.d_out), dim=-1)
        context_vectors = attn_weights @ values
        return context_vectors

class CausalAttention(nn.Module):
    """
    Causal Self-Attention: Restricts tokens to only attend to past and current tokens
    by applying an upper-triangular mask (torch.triu with diagonal=1) filled with -inf.
    """
    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # torch.triu with diagonal=1 has 1s on future positions (above main diagonal)
        # register_buffer ensures mask moves to GPU/MPS automatically with model.to(device)
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # Handles 3D batched inputs [batch_size, seq_len, d_in]
        queries = self.W_query(x)      # [b, num_tokens, d_out]
        keys    = self.W_key(x)        # [b, num_tokens, d_out]
        values  = self.W_value(x)      # [b, num_tokens, d_out]

        # 1. Compute raw scores: [b, num_tokens, d_out] @ [b, d_out, num_tokens] -> [b, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(-2, -1)

        # 2. Apply Causal Mask: Replace future token scores (where mask == 1) with -inf
        causal_mask = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(causal_mask, -torch.inf)


        # 3. Softmax Normalization (exp(-inf) = 0.0, future weights become 0%)
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.d_out), dim=-1)

        # 4. Apply Dropout to attention weights
        # Note: PyTorch's Inverted Dropout scales remaining non-zero weights by 1/(1-p) automatically,
        # preserving an expected sum of 1.0. No re-normalization is performed after dropout (running Softmax
        # again would turn dropped 0s back into exp(0)=1.0, destroying the dropout mask).
        attn_weights = self.dropout(attn_weights)


        # 5. Compute Context Vectors: [b, num_tokens, num_tokens] @ [b, num_tokens, d_out] -> [b, num_tokens, d_out]
        context_vectors = attn_weights @ values
        return context_vectors


def self_attention_class_demo():
    print("\n" + "=" * 70)
    print("PART 2: SELF-ATTENTION MODULES WITH TRAINABLE WEIGHTS (nn.Module)")
    print("=" * 70)

    # Sequence of 3 tokens, each with a 4-dimensional embedding vector
    inputs = torch.tensor([
        [0.43, 0.15, 0.89, 0.55],
        [0.55, 0.87, 0.66, 0.23],
        [0.57, 0.85, 0.08, 0.12]
    ], dtype=torch.float32)

    d_in = 4
    d_out = 2

    # Demo V1 (nn.Parameter)
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    context_v1 = sa_v1(inputs)
    print(f"\n1. SelfAttention_v1 (nn.Parameter):")
    print(f"   Input shape:   {list(inputs.shape)}")
    print(f"   Output shape:  {list(context_v1.shape)}")
    print(f"   Context vectors:\n{context_v1}")

    # Demo V2 (nn.Linear)
    torch.manual_seed(123)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    context_v2 = sa_v2(inputs)
    print(f"\n2. SelfAttention_v2 (nn.Linear):")
    print(f"   Input shape:   {list(inputs.shape)}")
    print(f"   Output shape:  {list(context_v2.shape)}")
    print(f"   Context vectors:\n{context_v2}")

    # 3D Batched Input Demo (Batch size 2, Sequence length 3, Embedding dim 4)
    batch_inputs = torch.stack([inputs, inputs * 1.5])
    context_batch = sa_v2(batch_inputs)
    print(f"\n3. SelfAttention_v2 3D Batched Input Test:")
    print(f"   Batch Input shape:  {list(batch_inputs.shape)}  [batch_size, seq_len, d_in]")
    print(f"   Batch Output shape: {list(context_batch.shape)}  [batch_size, seq_len, d_out]")

    # 4. Weight Transfer Verification (sa_v2 -> sa_v1)
    # PyTorch nn.Linear stores weight matrix as (d_out, d_in).
    # SelfAttention_v1 expects weight matrix as (d_in, d_out).
    sa_v1.W_Q.data = sa_v2.W_query.weight.T.clone()
    sa_v1.W_K.data = sa_v2.W_key.weight.T.clone()
    sa_v1.W_V.data = sa_v2.W_value.weight.T.clone()

    context_v1_transferred = sa_v1(inputs)
    print(f"\n4. Weight Transfer Verification (sa_v2 -> sa_v1):")
    print(f"   Transferred sa_v1 Context Vectors:\n{context_v1_transferred}")
    print(f"   sa_v2 Context Vectors:\n{context_v2}")
    # 5. CausalAttention Demo
    torch.manual_seed(123)
    causal_attn = CausalAttention(d_in=4, d_out=2, context_length=1024, dropout=0.0)
    context_causal = causal_attn(batch_inputs)
    print(f"\n5. CausalAttention (Masked Causal Self-Attention):")
    print(f"   Batch Input shape:  {list(batch_inputs.shape)}  [batch_size, seq_len, d_in]")
    print(f"   Batch Output shape: {list(context_causal.shape)}  [batch_size, seq_len, d_out]")
    print(f"   Context vectors:\n{context_causal}")
    print("=" * 70)


if __name__ == "__main__":
    simplified_attention_demo()
    self_attention_class_demo()



