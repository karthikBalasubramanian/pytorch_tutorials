from typing_extensions import Dict
import os
import re
import urllib.request


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