# coding=utf-8
"""
Setup external Hindi NER datasets for transfer learning:
1. WikiAnn Hindi NER - from HuggingFace
2. IIT Bombay Hindi NER - from research repository

Transfer Learning approach:
1. Pre-train on large Hindi NER dataset
2. Fine-tune on Hinglish data
This typically improves low-resource performance significantly
"""

def setup_wikiann_dataset():
    """
    Download WikiAnn Hindi NER dataset from HuggingFace
    
    WikiAnn contains ~30K Hindi NER annotations (PER, LOC, ORG)
    This provides much more training data than our ~1K Hinglish sentences
    """
    print("\n" + "="*70)
    print("TRANSFER LEARNING: Setting up WikiAnn Hindi NER")
    print("="*70)
    
    print("\nOption 1: Automatic Download (Recommended)")
    print("-"*70)
    print("""
from datasets import load_dataset

# Load WikiAnn Hindi dataset
dataset = load_dataset('wikiann', 'hi')

# Save to file
with open('src/data/hindi_wikiann.train', 'w', encoding='utf-8') as f:
    for example in dataset['train']:
        tokens = example['tokens']
        tags = example['ner_tags']
        for token, tag_id in zip(tokens, tags):
            tag_name = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'][tag_id]
            f.write(f"{token}\\t{tag_name}\\n")
        f.write("\\n")

print("WikiAnn Hindi dataset saved!")
    """)
    
    print("\nOption 2: Manual Download")
    print("-"*70)
    print("""
1. Go to: https://huggingface.co/datasets/wikiann
2. Download Hindi split
3. Convert to BIO format (word TAB tag)
4. Save to: src/data/hindi_wikiann.train
    """)
    
    print("\nOption 3: Use Pre-converted Dataset")
    print("-"*70)
    print("""
# Alternative Hindi NER datasets:
- Hindi Wikipedia Named Entity Recognition (Kaggle)
- IIT-Bombay Hindi NER Dataset

These can be preprocessed and added to training data.
    """)

def setup_transfer_learning_pipeline():
    """
    Create transfer learning training strategy
    """
    print("\n" + "="*70)
    print("TRANSFER LEARNING PIPELINE")
    print("="*70)
    
    strategy = """
STEP 1: Pre-training on Hindi NER
- Train on WikiAnn Hindi (30K+ sentences)
- Learn Hindi linguistic patterns
- Learn entity types (PER, ORG, LOC)
- Save checkpoint: models/hindi_pretrain.pt

STEP 2: Fine-tuning on Hinglish
- Load pre-trained weights
- Train on Hinglish data (1.2K sentences)
- Adapt to code-mixed patterns
- Save model: models/hinglish_ner.pt

EXPECTED IMPROVEMENTS:
- ORG F1: 12-15% improvement (from weak baseline)
- Overall F1: 5-10% improvement
- Reduced overfitting on small dataset
- Better generalization

IMPLEMENTATION:
1. Add --pretrain flag to train.py
2. Load pre-trained model weights
3. Continue training with lower learning rate (0.005 instead of 0.015)
4. Use all data for fine-tuning (no early stopping in pre-train)
    """
    
    print(strategy)

def create_training_script():
    """
    Script to run transfer learning
    """
    script = """
# STEP 1: Pre-train on Hindi
python src/train.py \\
    --name hindi_pretrain \\
    --train src/data/hindi_wikiann.train \\
    --dev src/data/hindi_wikiann.dev \\
    --test src/data/hindi_wikiann.test \\
    --tag_scheme bio

# STEP 2: Fine-tune on Hinglish (transfer learning)
python src/train.py \\
    --name hinglish_ner \\
    --train src/data/hineng.train \\
    --dev src/data/hineng.valid \\
    --test src/data/hineng.test \\
    --reload 1 \\
    --word_dim 300
    """
    
    print("\n" + "="*70)
    print("TRANSFER LEARNING COMMANDS")
    print("="*70)
    print(script)

if __name__ == "__main__":
    setup_wikiann_dataset()
    setup_transfer_learning_pipeline()
    create_training_script()
    
    print("\n" + "="*70)
    print("TRANSFER LEARNING SETUP COMPLETE")
    print("="*70)
    print("\nNEXT STEPS:")
    print("1. Download WikiAnn Hindi dataset (or use provided link)")
    print("2. Convert to BIO format")
    print("3. Run pre-training on Hindi NER")
    print("4. Run fine-tuning on Hinglish data")
    print("="*70 + "\n")
