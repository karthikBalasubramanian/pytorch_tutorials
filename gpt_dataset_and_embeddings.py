from networkx.algorithms.traversal import breadth_first_search
import os
import urllib.request
import urllib

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

class GPTDatasetV1(Dataset):
    """
    PyTorch Dataset for preparing input-target chunk pairs for autoregressive GPT model training.

    Each dataset item is a tuple of 1D integer tensors: (input_chunk, target_chunk)
        - input_chunk shape:  [sequence_length]  (where sequence_length = max_length)
        - target_chunk shape: [sequence_length]  (input sequence shifted right by 1 token)
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_data_loader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    Creates a PyTorch DataLoader for tokenized text chunks.

    Tensor Dimensions Breakdown:
    ----------------------------
    1. DataLoader Batch Output:
       - input_ids:  [batch_size, sequence_length]
       - target_ids: [batch_size, sequence_length]
       (e.g., [8, 4] for 8 batch samples, each 4 tokens long)

    2. Token Embeddings Output (after passing input_ids into nn.Embedding(vocab_size, embedding_dim)):
       - Shape: [batch_size, sequence_length, embedding_dim]
       
       Dimensions:
         - Dimension 0 (batch_size):      Number of text sequences processed in parallel (e.g., 8)
         - Dimension 1 (sequence_length): Context length / max_length of tokens per sequence (e.g., 4)
         - Dimension 2 (embedding_dim):   Feature vector size per token (e.g., 768 for GPT-2 Small)

       Note: `vocab_size` (e.g., 50,257) is the size of the embedding lookup table row dimension, 
             which is indexed away during lookup and does NOT appear in the final tensor shape.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return data_loader

if __name__ == "__main__":
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    file_path = "the-verdict.txt"
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    data_loader = create_data_loader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    print("Number of batches:", len(data_loader))

    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    print(f"Input shape (batch_size, sequence_length): {first_batch[0].shape}")
    print(f"Target shape (batch_size, sequence_length): {first_batch[1].shape}")

    input_example, target_example = first_batch 

    print(f"Input IDs:\n{input_example}")
    print(f"Target IDs:\n{target_example}")

    for i in range(len(input_example)):
        print(f"{input_example[i]} -> {target_example[i]}")

    # Demonstration of 3D Token Embedding shape transformation:
    # nn.Embedding(vocab_size=50257, embedding_dim=768)
    vocab_size = 50257
    embedding_dim = 768
    token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

    # Input: [batch_size, sequence_length] -> Output: [batch_size, sequence_length, embedding_dim]
    embedded_inputs = token_embedding_layer(first_batch[0])
    b, s, d = embedded_inputs.shape
    print(f"\nCreated an {b}x{s}x{d} embedding tensor. Now let's decode it back:")

    # Note on Positional Embeddings:
    # -----------------------------
    # We can decode directly back to original text here WITHOUT positional embeddings because
    # decoding relies ONLY on token identity (which is position-independent; e.g. Token 3797 is always "cat").
    # Positional embeddings are needed during LLM generation / self-attention to understand 
    # word order and relative distance ("Dog bites man" vs "Man bites dog").

    # 1. Un-embed / Project from embedding_dim (768) back to vocab_size (50257) logits
    logits = embedded_inputs @ token_embedding_layer.weight.T  # Shape: [batch_size, sequence_length, vocab_size]
    print(f"Logits shape: {logits.shape}  # [batch_size, sequence_length, vocab_size]")

    # 2. Extract Token IDs by argmax over vocabulary dimension (-1)
    decoded_token_ids = torch.argmax(logits, dim=-1)  # Shape: [batch_size, sequence_length]
    print(f"Decoded token IDs shape: {decoded_token_ids.shape}  # [batch_size, sequence_length]")

    # Verify that un-embedding recovers original input IDs
    matches_original = torch.equal(first_batch[0], decoded_token_ids)
    print(f"Matches original input IDs exactly: {matches_original}")

    # 3. Decode Token IDs to String text per batch sequence using tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    decoded_texts = [tokenizer.decode(seq.tolist()) for seq in decoded_token_ids]
    print(f"\nDecoded batch text sequences:")
    for idx, text in enumerate(decoded_texts):
        print(f"  Batch item {idx}: '{text}'")

    # Demonstration of Positional Embeddings:
    # ---------------------------------------
    # Token embeddings (embedded_inputs) only carry word identity, but Transformer Self-Attention
    # is order-blind (permutation invariant). To give the model awareness of token order 
    # (word #0, word #1, word #2, etc.), we add positional embeddings to the token embeddings.

    # Distinction between context_length and sequence_length (s):
    # - context_length (256): Max context capacity of the model (positional table size: positions 0..255).
    # - sequence_length / s (4): Actual number of tokens in the current batch (first_batch[0].shape[1]).
    # We retrieve positional vectors for positions 0..s-1 (the exact sequence length of the current batch).

    context_length = 256  # Model's max supported context window capacity
    pos_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)

    # 1. Create position IDs for current sequence length `s` (4): tensor([0, 1, 2, 3])
    pos_ids = torch.arange(s)  # Shape: [sequence_length = 4]

    # 2. Look up positional vectors for positions 0..s-1: Shape [sequence_length, embedding_dim] -> [4, 768]
    pos_embeddings = pos_embedding_layer(pos_ids)

    # 3. Combine Token Embeddings + Positional Embeddings via elementwise addition:
    #    [batch_size, sequence_length, embedding_dim] + [sequence_length, embedding_dim]
    #    PyTorch broadcasts [4, 768] across the batch dimension to match [8, 4, 768].
    input_embeddings = embedded_inputs + pos_embeddings

    print(f"\nDemonstration of Positional Embedding combination:")
    print(f"Current batch sequence length (s): {s}")
    print(f"Positional embedding shape: {pos_embeddings.shape}  # [sequence_length, embedding_dim]")
    print(f"Final combined input_embeddings shape: {input_embeddings.shape}  # [batch_size, sequence_length, embedding_dim]")
    print("Success: Combined embeddings now contain both token semantic identity AND word sequence order!")






