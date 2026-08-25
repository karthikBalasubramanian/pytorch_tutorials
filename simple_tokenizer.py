from typing_extensions import Dict
import os
import re
import urllib.request
import tiktoken



class SimpleTokenizerV1:
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.int_to_str[i] for i in ids]
        text = " ".join(tokens)
        # Remove whitespace before punctuation marks
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2(SimpleTokenizerV1):
    """
    Improved tokenizer that handles out-of-vocabulary (OOV) tokens <|unk|> and adds a special context token for 
    end of text "<|endoftext|>" to the vocabulary.
    """
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids




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

    # Preprocess raw_text to extract all unique word and punctuation tokens
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_tokens = sorted(set(preprocessed))

    # Create vocabulary mapping tokens to integer IDs
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    print("\n============================================\n")
    print(f"Vocabulary size: {len(vocab)} unique tokens")

    # Instantiate tokenizer
    tokenizer = SimpleTokenizerV1(vocab)

    # Test encoding and decoding (using text present in the-verdict.txt)
    sample_text = """"I HAD always thought Jack Gisburn rather a cheap genius"--so it was no great surprise to me"""
    encoded_ids = tokenizer.encode(sample_text)
    print("\nSample Text:\n", sample_text)
    print("\nEncoded Token IDs:\n", encoded_ids)
    
    decoded_text = tokenizer.decode(encoded_ids)
    print("\nDecoded Text:\n", decoded_text)

    # Negative Case: Out-of-Vocabulary (OOV) text containing unseen words
    unseen_text = "Hello, how are you?"
    print("\n--- Negative Case: Out-of-Vocabulary (OOV) Text ---")
    print("Unseen Input Text:", unseen_text)
    try:
        tokenizer.encode(unseen_text)
    except KeyError as e:
        print(f"KeyError caught: Token {e} is NOT in the vocabulary!")
        print("--> SimpleTokenizerV1 fails on words that were absent from the training corpus.")

    
    all_tokens_v2 = all_tokens + ["<|unk|>", "<|endoftext|>"]
    vocab_v2 = {token: integer for integer, token in enumerate(all_tokens_v2)}
    tokenizer_v2 = SimpleTokenizerV2(vocab_v2)
    print("\n============================================\n")
    print("Vocabulary size v2:", len(vocab_v2))
    
    # Add a space before <|endoftext|> so regex splits it as a separate token from 'me'
    sample_text_v2 = sample_text + " <|endoftext|>"
    encoded_ids_v2 = tokenizer_v2.encode(sample_text_v2)
    print("Encoded Token IDs (v2):\n", encoded_ids_v2)

    decoded_text_v2 = tokenizer_v2.decode(encoded_ids_v2)
    print("Decoded Text (v2):\n", decoded_text_v2)

    # unseen text with tokenizer_v2
    print("\n--- Case: Unseen Text (with tokenizer_v2) ---")
    print("Unseen Input Text:", unseen_text)
    encoded_ids_v2 = tokenizer_v2.encode(unseen_text)
    print("\nEncoded Token IDs (v2):\n", encoded_ids_v2)
    decoded_text_v2 = tokenizer_v2.decode(encoded_ids_v2)
    print("Decoded Text (v2):\n", decoded_text_v2)

    # BPE Tokenizer Example using OpenAI's tiktoken (GPT-2 vocabulary)
    print("\n============================================\n")
    print("--- BPE Tokenizer Example (tiktoken gpt2) ---")
    tokenizer_bpe = tiktoken.get_encoding("gpt2")
    
    text_bpe = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces "
        "of someunknownPlace."
    )
    
    # -----------------------------------------------------------------------------
    # SECURITY FEATURE & SPECIAL TOKEN PROTECTION:
    # -----------------------------------------------------------------------------
    # By default, tiktoken raises a ValueError if special control tokens like
    # `<|endoftext|>` appear in raw input strings. This is a security measure to
    # prevent "Special Token Injection Attacks" (where untrusted user text might
    # inject control signals to hijack prompt boundaries or system instructions).
    #
    # Passing `allowed_special={"<|endoftext|>"}` explicitly opts-in to parsing
    # specified special control tokens for trusted internal dataset pipelines.
    # -----------------------------------------------------------------------------
    integers = tokenizer_bpe.encode(text_bpe, allowed_special={"<|endoftext|>"})
    print("Encoded Token IDs (tiktoken):\n", integers)
    print(f"Max Token ID in gpt2 vocab: {tokenizer_bpe.max_token_value}")
    print("--> ID 50256 corresponds to <|endoftext|> (index 50256 in 50,257 total vocab)")
    
    decoded_bpe = tokenizer_bpe.decode(integers)
    print("\nDecoded Text (tiktoken):\n", decoded_bpe)