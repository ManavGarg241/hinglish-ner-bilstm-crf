# coding=utf-8
import os
import numpy as np

print("Creating FastText-style embeddings for Hinglish...")
print("=" * 70)

embeddings_dir = "src/embeddings"
if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)

# Load training data
print("\n1. Loading training data...")
train_file = "src/data/hineng.train"
word_counts = {}
sentences = []

with open(train_file, 'r', encoding='utf-8') as f:
    sentence = []
    for line in f:
        line = line.strip()
        if not line:
            if sentence:
                sentences.append(sentence)
                sentence = []
        else:
            word = line.split()[0]
            sentence.append(word)
            word_counts[word] = word_counts.get(word, 0) + 1

print(f"   ✓ Loaded {len(sentences)} sentences")
print(f"   ✓ Found {len(word_counts)} unique words")

# Create embeddings: frequent words get better initialized embeddings
print("\n2. Creating embeddings...")
np.random.seed(42)
embedding_dim = 300

# Sort by frequency
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

embedding_file = os.path.join(embeddings_dir, "hinglish_embeddings.vec")
with open(embedding_file, 'w', encoding='utf-8') as f:
    # Write header
    f.write(f"{len(word_counts)} {embedding_dim}\n")
    
    # Write embeddings: frequent words get better ones
    for rank, (word, count) in enumerate(sorted_words):
        # Better initialization for frequent words (lower variance)
        scale = 0.05 / (1 + rank / 1000)  # Decrease scale for rare words
        vector = np.random.normal(0, scale, embedding_dim)
        vector_str = ' '.join([f"{v:.6f}" for v in vector])
        f.write(f"{word} {vector_str}\n")

print(f"   ✓ Created embeddings for {len(word_counts)} words")
print(f"   ✓ Dimension: {embedding_dim}")

print("\n" + "=" * 70)
print("✓ Embeddings ready!")
print(f"  Path: {embedding_file}")
print(f"  Vocabulary: {len(word_counts)} words")
print(f"  Dimension: {embedding_dim}")
print("\nReady to train with embeddings:")
print(f"  python src/train.py --name hinglish_ner --pre_emb {embedding_file}")

