# coding=utf-8
import torch
import pickle
import numpy as np
from torch.autograd import Variable
from model import BiLSTM_CRF
import loader

def load_model_and_mappings(model_path, mapping_path):
    """Load trained model and word/char/tag mappings"""
    # Load mappings
    with open(mapping_path, 'rb') as f:
        mappings = pickle.load(f)
    
    word_to_id = mappings['word_to_id']
    char_to_id = mappings['char_to_id']
    tag_to_id = mappings['tag_to_id']
    parameters = mappings['parameters']
    word_embeds = mappings['word_embeds']
    
    # Create reverse mappings
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    return model, word_to_id, char_to_id, tag_to_id, id_to_tag, parameters, device

def preprocess_text(text, word_to_id, char_to_id, lower=True, zeros=False):
    """Convert text to model input format"""
    words = text.strip().split()
    
    # Convert words to IDs
    word_ids = []
    char_ids = []
    
    for word in words:
        if lower:
            word = word.lower()
        if zeros:
            word = word.replace('0', '0')
        
        # Get word ID (use UNK for unknown words)
        word_id = word_to_id.get(word, word_to_id.get('UNK', 0))
        word_ids.append(word_id)
        
        # Get character IDs
        chars = []
        for char in word:
            char_id = char_to_id.get(char, char_to_id.get('UNK', 0))
            chars.append(char_id)
        char_ids.append(chars)
    
    return words, word_ids, char_ids

def predict_tags(text, model, word_to_id, char_to_id, id_to_tag, parameters, device):
    """Predict tags for input text"""
    words, word_ids, char_ids = preprocess_text(
        text, word_to_id, char_to_id, 
        lower=parameters['lower'], 
        zeros=parameters['zeros']
    )
    
    if len(words) == 0:
        return []
    
    # Prepare character mask (same as training)
    if parameters['char_mode'] == 'CNN':
        chars_length = [len(c) for c in char_ids]
        char_maxl = max(chars_length) if chars_length else 1
        chars_mask = np.zeros((len(chars_length), char_maxl), dtype='int')
        for i, c in enumerate(char_ids):
            chars_mask[i, :chars_length[i]] = c
        chars_mask = Variable(torch.LongTensor(chars_mask))
        d = {}
    
    # Prepare tensors
    dwords = Variable(torch.LongTensor(word_ids))
    dcaps = Variable(torch.LongTensor([0] * len(word_ids)))  # No capitalization feature
    
    # Run model
    if torch.cuda.is_available():
        _, pred_tags = model(dwords.cuda(), chars_mask.cuda(), dcaps.cuda(), chars_length, d)
    else:
        _, pred_tags = model(dwords, chars_mask, dcaps, chars_length, d)
    
    # Convert tag IDs to tag names
    tag_names = [id_to_tag.get(tag_id, 'O') for tag_id in pred_tags]
    
    return words, tag_names

def print_predictions(words, tags):
    """Pretty print word-tag pairs"""
    print("\n" + "="*60)
    print("NER Predictions:")
    print("="*60)
    for word, tag in zip(words, tags):
        print(f"{word:20} -> {tag}")
    print("="*60 + "\n")

def main():
    # Load model and mappings
    model_path = "models/hinglish_ner"
    mapping_path = "models/mapping.pkl"
    
    print("Loading model and mappings...")
    model, word_to_id, char_to_id, tag_to_id, id_to_tag, parameters, device = \
        load_model_and_mappings(model_path, mapping_path)
    print(f"Model loaded! Device: {device}")
    print(f"Total tags: {len(tag_to_id)}")
    print(f"Total words in vocabulary: {len(word_to_id)}\n")
    
    # Interactive inference
    print("Enter Hinglish text to get NER predictions (type 'quit' to exit):")
    print("-" * 60)
    
    while True:
        text = input("\nEnter text: ").strip()
        
        if text.lower() == 'quit':
            print("Exiting...")
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        try:
            words, tags = predict_tags(text, model, word_to_id, char_to_id, id_to_tag, parameters, device)
            print_predictions(words, tags)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
