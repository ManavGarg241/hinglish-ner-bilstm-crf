# coding=utf-8
"""
Download and combine multilingual FastText embeddings (English + Hindi)
for improved Hinglish NER performance
"""
import os
import sys
import urllib.request
from pathlib import Path

# Create embeddings directory if not exists
emb_dir = "src/embeddings"
Path(emb_dir).mkdir(parents=True, exist_ok=True)

print("Downloading multilingual FastText embeddings...")
print("=" * 70)

# URLs for FastText embeddings (official Facebook Research)
urls = {
    'english': 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz',
    'hindi': 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.vec.gz',
}

# Since downloading large files takes time, provide alternative approach
print("Note: FastText embedding files are ~700MB each (compressed)")
print("Downloading will take 5-10 minutes per file.\n")

print("OPTION 1: Automatic Download (takes ~20 minutes)")
print("-" * 70)

import gzip
import shutil

for lang, url in urls.items():
    gz_path = os.path.join(emb_dir, f'cc.{lang}.300.vec.gz')
    vec_path = os.path.join(emb_dir, f'cc.{lang}.300.vec')
    
    if os.path.isfile(vec_path):
        print(f"[OK] {vec_path} already exists, skipping...")
        continue
    
    try:
        print(f"\nDownloading cc.{lang}.300.vec.gz...")
        urllib.request.urlretrieve(url, gz_path)
        print(f"[OK] Downloaded to {gz_path}")
        
        print(f"Extracting {gz_path}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(vec_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"[OK] Extracted to {vec_path}")
        
        # Remove gz file to save space
        os.remove(gz_path)
        print(f"[OK] Removed {gz_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to download {lang}: {e}")
        print(f"\nOPTION 2: Manual Download")
        print("-" * 70)
        print(f"1. Download from: {url}")
        print(f"2. Extract .gz file to: {vec_path}")
        print(f"3. Run this script again\n")

print("\n" + "=" * 70)
print("Embedding download/setup complete!")
print(f"English embeddings: {os.path.join(emb_dir, 'cc.en.300.vec')}")
print(f"Hindi embeddings:   {os.path.join(emb_dir, 'cc.hi.300.vec')}")
print("=" * 70)
