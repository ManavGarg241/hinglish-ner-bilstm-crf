# coding=utf-8
import re
import os
from pathlib import Path

def clean_word(word):
    """Clean a word by removing social media noise"""
    
    # Remove URLs (http, https, www, shortened links)
    if re.match(r'^https?://', word) or re.match(r'^www\.', word) or re.match(r'^bit\.ly', word):
        return None
    
    # Remove @mentions but keep the text after @
    if word.startswith('@'):
        cleaned = re.sub(r'^@', '', word)
        if len(cleaned) > 0:
            return cleaned
        return None
    
    # Handle hashtags - keep the word part, remove #
    if word.startswith('#'):
        cleaned = re.sub(r'^#+', '', word)
        if len(cleaned) > 0:
            return cleaned
        return None
    
    # Remove remaining URLs in middle of text
    if 'http' in word.lower() or '.com' in word.lower() or '.co' in word.lower():
        return None
    
    return word

def clean_data_file(input_file, output_file):
    """Clean a CoNLL format data file"""
    
    print(f"Cleaning {input_file}...")
    
    original_lines = 0
    removed_lines = 0
    removed_sentences = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    cleaned_lines = []
    current_sentence_has_content = False
    sentence_buffer = []
    
    for line in lines:
        original_lines += 1
        line = line.rstrip('\n')
        
        if not line.strip():  # Empty line (sentence boundary)
            if current_sentence_has_content:
                cleaned_lines.extend(sentence_buffer)
                cleaned_lines.append('')
            else:
                if sentence_buffer:  # Had content but all removed
                    removed_sentences += 1
            
            sentence_buffer = []
            current_sentence_has_content = False
            continue
        
        # Parse word and tag
        parts = line.split()
        if len(parts) < 2:
            continue
        
        word = parts[0]
        tag = parts[1]
        
        # Clean the word
        cleaned_word = clean_word(word)
        
        if cleaned_word is None:
            removed_lines += 1
        else:
            sentence_buffer.append(f"{cleaned_word} {tag}")
            current_sentence_has_content = True
    
    # Handle last sentence if file doesn't end with blank line
    if current_sentence_has_content:
        cleaned_lines.extend(sentence_buffer)
        cleaned_lines.append('')
    
    # Write cleaned file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in cleaned_lines:
            f_out.write(line + '\n')
    
    print(f"  ✓ Original lines: {original_lines}")
    print(f"  ✓ Removed noisy tokens: {removed_lines}")
    print(f"  ✓ Removed empty sentences: {removed_sentences}")
    print(f"  ✓ Output: {output_file}\n")
    
    return removed_lines, removed_sentences

# Clean all data files
print("=" * 70)
print("CLEANING SOCIAL MEDIA NOISE FROM DATA")
print("=" * 70)
print()

files_to_clean = [
    ("src/data/hineng.train", "src/data/hineng.train"),
    ("src/data/hineng.valid", "src/data/hineng.valid"),
    ("src/data/hineng.test", "src/data/hineng.test"),
]

total_removed = 0
total_empty_sentences = 0

for input_file, output_file in files_to_clean:
    if os.path.exists(input_file):
        removed, empty_sents = clean_data_file(input_file, output_file)
        total_removed += removed
        total_empty_sentences += empty_sents

print("=" * 70)
print("CLEANUP COMPLETE")
print("=" * 70)
print(f"Total noisy tokens removed: {total_removed}")
print(f"Total empty sentences removed: {total_empty_sentences}")
print()
print("Data is now cleaned and ready for training with embeddings!")
print("Next: python src/train.py --name hinglish_ner")
