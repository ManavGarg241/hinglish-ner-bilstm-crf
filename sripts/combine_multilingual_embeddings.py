# coding=utf-8
"""
Combine English and Hindi FastText embeddings for Hinglish
This takes vocabulary from training data and pulls embeddings from both cc.en and cc.hi
"""
import os
import numpy as np
from pathlib import Path

def load_embeddings_dict(emb_file, max_lines=None, verbose=True):
    """Load FastText embeddings into dictionary"""
    embeddings = {}
    emb_dim = None
    
    try:
        with open(emb_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip().split()
            
            # Check if first line is header (vocab_size dim)
            if len(first_line) == 2:
                emb_dim = int(first_line[1])
                start_iter = f
            else:
                # First line is actual data
                emb_dim = len(first_line) - 1
                word = first_line[0]
                vector = np.array([float(v) for v in first_line[1:]], dtype=np.float32)
                embeddings[word] = vector
                start_iter = f
            
            count = 0
            for line in start_iter:
                if max_lines and count >= max_lines:
                    break
                
                parts = line.strip().split()
                if len(parts) == emb_dim + 1:
                    word = parts[0]
                    vector = np.array([float(v) for v in parts[1:]], dtype=np.float32)
                    embeddings[word] = vector
                    count += 1
                
                if verbose and count % 100000 == 0:
                    print(f"  Loaded {count} embeddings from {emb_file}")
        
        if verbose:
            print(f"[OK] Loaded {len(embeddings)} embeddings ({emb_dim}D) from {emb_file}")
        
        return embeddings, emb_dim
    except Exception as e:
        print(f"[ERROR] Failed to load {emb_file}: {e}")
        return {}, None

def load_vocab_from_data(data_file):
    """Load vocabulary from training data"""
    vocab = set()
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if parts:
                        word = parts[0].lower()
                        vocab.add(word)
    except Exception as e:
        print(f"[ERROR] Failed to load vocab from {data_file}: {e}")
    
    return vocab

def combine_embeddings():
    """Combine EN + HI FastText embeddings for Hinglish"""
    emb_dir = "src/embeddings"
    en_file = os.path.join(emb_dir, "cc.en.300.vec")
    hi_file = os.path.join(emb_dir, "cc.hi.300.vec")
    output_file = os.path.join(emb_dir, "hinglish_multilingual.vec")
    data_file = "src/data/hineng.train"
    
    print("\n" + "=" * 70)
    print("Combining multilingual FastText embeddings (EN + HI)")
    print("=" * 70)
    
    # Check if files exist
    if not os.path.isfile(en_file):
        print(f"[!] {en_file} not found")
        print(f"    Run: python download_multilingual_fasttext.py")
        return False
    
    if not os.path.isfile(hi_file):
        print(f"[!] {hi_file} not found")
        print(f"    Run: python download_multilingual_fasttext.py")
        return False
    
    # Load vocabulary from training data
    print(f"\nLoading vocabulary from {data_file}...")
    vocab = load_vocab_from_data(data_file)
    print(f"[OK] Vocabulary size: {len(vocab)} unique words")
    
    # Load embeddings (sample to speed up for testing)
    print(f"\nLoading English embeddings (sampling for speed)...")
    en_embeds, en_dim = load_embeddings_dict(en_file, max_lines=300000)
    
    print(f"\nLoading Hindi embeddings (sampling for speed)...")
    hi_embeds, hi_dim = load_embeddings_dict(hi_file, max_lines=300000)
    
    if not en_embeds or not hi_embeds:
        print("[ERROR] Failed to load embeddings")
        return False
    
    # Create combined embeddings
    print(f"\nCombining embeddings for vocabulary...")
    combined = {}
    matched_en = 0
    matched_hi = 0
    matched_both = 0
    
    for word in vocab:
        has_en = word in en_embeds
        has_hi = word in hi_embeds
        
        if has_en and has_hi:
            # Average EN and HI embeddings for better representation
            combined[word] = (en_embeds[word] + hi_embeds[word]) / 2.0
            matched_both += 1
        elif has_en:
            combined[word] = en_embeds[word]
            matched_en += 1
        elif has_hi:
            combined[word] = hi_embeds[word]
            matched_hi += 1
    
    print(f"\nEmbedding Coverage:")
    print(f"  Found in BOTH EN+HI: {matched_both}")
    print(f"  Found in EN only:    {matched_en}")
    print(f"  Found in HI only:    {matched_hi}")
    print(f"  Total coverage:      {len(combined)}/{len(vocab)} ({100*len(combined)/len(vocab):.1f}%)")
    
    # Save combined embeddings
    print(f"\nSaving combined embeddings to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"{len(combined)} {en_dim}\n")
            
            # Write embeddings
            for word, vector in combined.items():
                vec_str = ' '.join([str(v) for v in vector])
                f.write(f"{word} {vec_str}\n")
        
        print(f"[OK] Saved {len(combined)} embeddings to {output_file}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save embeddings: {e}")
        return False

if __name__ == "__main__":
    success = combine_embeddings()
    if success:
        print("\n" + "=" * 70)
        print("Update train.py to use: src/embeddings/hinglish_multilingual.vec")
        print("=" * 70)
    else:
        print("\n[!] Combination failed. Download FastText embeddings first.")
