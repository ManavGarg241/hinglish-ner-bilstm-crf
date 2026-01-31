# coding=utf-8
"""
Generate synthetic Hinglish sentences with ORG entities for data augmentation
This addresses the limited ORGANISATION data (only 285 tags in original dataset)
"""
import random

# Common Hinglish company/organization patterns
# Format: (prefix_template, org_name_hindi, org_name_english, suffix_template)
org_patterns = [
    # Company establishment
    ("", "Google", "B-ORG", " ne Delhi me office khola"),
    ("", "Microsoft", "B-ORG", " ne Mumbai me naya branch setup kiya"),
    ("", "Amazon", "B-ORG", " ne Bangalore me data center bna raha hai"),
    ("", "Facebook", "B-ORG", " ne India me naya team hire kar raha hai"),
    ("", "Apple", "B-ORG", " ne Pune me manufacturing facility kholi"),
    ("", "IBM", "B-ORG", " ne Hyderabad me IT center expand kiya"),
    ("", "Intel", "B-ORG", " ne Gurugram me research lab khola"),
    ("", "TCS", "B-ORG", " ne London me office set up kiya"),
    ("", "Infosys", "B-ORG", " ne Singapore me expansion kiya"),
    ("", "Wipro", "B-ORG", " ne Frankfurt me naya center khola"),
    
    # Acquisitions/partnerships
    ("", "Flipkart", "B-ORG", " ne Myntra ko acquire kiya"),
    ("", "Ola", "B-ORG", " ne Uber ke saath deal ki baat chal rahi hai"),
    ("", "Paytm", "B-ORG", " ne MobiKwik ke saath partnership kiya"),
    ("", "Zomato", "B-ORG", " ne Uber Eats ko khareed liya"),
    ("", "Swiggy", "B-ORG", " ne Supr Daily ko integrate kiya"),
    
    # Government/Public sectors
    ("", "ISRO", "B-ORG", " ne Chandrayaan mission launch kiya"),
    ("", "Railway", "B-ORG", " ne naya express train chalaya"),
    ("", "NTPC", "B-ORG", " ne renewable energy project start kiya"),
    ("", "Air India", "B-ORG", " ne naye aircraft order diye"),
    ("", "BSNL", "B-ORG", " ne 4G network expansion plan announce kiya"),
    
    # Banks
    ("", "HDFC", "B-ORG", " ne mortgage rates badl diye"),
    ("", "ICICI", "B-ORG", " ne digital banking service launch ki"),
    ("", "SBI", "B-ORG", " ne naya saving account scheme nikala"),
    ("", "Axis Bank", "B-ORG", " ne credit card limits increase kiye"),
    
    # News/Media
    ("", "Times of India", "B-ORG", " ne breaking news publish ki"),
    ("", "NDTV", "B-ORG", " ne prime time debate schedule kiya"),
    ("", "India Today", "B-ORG", " ne special investigation kiya"),
    ("", "BBC", "B-ORG", " ne documentary release ki"),
    
    # Clothing/Consumer
    ("", "Nike", "B-ORG", " ne naya shoe collection launch kiya"),
    ("", "Adidas", "B-ORG", " ne sports event sponsor kiya"),
    ("", "H&M", "B-ORG", " ne India me naya store khola"),
    ("", "Zara", "B-ORG", " ne summer collection release kiya"),
    
    # Auto industry
    ("", "Maruti", "B-ORG", " ne EV model announce kiya"),
    ("", "Hyundai", "B-ORG", " ne warranty extension kiya"),
    ("", "Tata Motors", "B-ORG", " ne electric bus launch kiya"),
    ("", "Mahindra", "B-ORG", " ne new SUV model introduce kiya"),
    
    # Education
    ("", "IIT", "B-ORG", " ne entrance exam conduct kiya"),
    ("", "Delhi University", "B-ORG", " ne admission process start kiya"),
    ("", "NIT", "B-ORG", " ne placement record set kiya"),
    ("", "IGNOU", "B-ORG", " ne online courses expand kiye"),
]

def generate_augmented_data(num_samples=250):
    """Generate synthetic Hinglish ORG sentences"""
    sentences = []
    
    for _ in range(num_samples):
        pattern = random.choice(org_patterns)
        prefix, org_name, org_tag, suffix = pattern
        
        # Create sentence
        sentence = f"{prefix}{org_name}{suffix}"
        
        # Tokenize and tag
        tokens = sentence.split()
        tags = []
        
        # Find and tag the ORG entity
        found = False
        for token in tokens:
            if org_name in token and not found:
                tags.append("B-ORG")
                found = True
            else:
                tags.append("O")
        
        # Store sentence and tags
        sentences.append((tokens, tags))
    
    return sentences

def write_augmented_data(sentences, output_path):
    """Write augmented data in BIO format"""
    with open(output_path, 'a', encoding='utf-8') as f:
        for tokens, tags in sentences:
            for token, tag in zip(tokens, tags):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")  # Sentence separator
    
    print(f"[OK] Added {len(sentences)} synthetic sentences to {output_path}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DATA AUGMENTATION: Generating Synthetic Hinglish ORG Sentences")
    print("="*70)
    
    # Generate synthetic data
    print("\nGenerating 250 synthetic ORG sentences...")
    augmented = generate_augmented_data(num_samples=250)
    
    # Show sample
    print("\nSample synthetic sentences:")
    for i, (tokens, tags) in enumerate(augmented[:5]):
        print(f"  {i+1}. {' '.join(tokens)}")
        print(f"     Tags: {tags}\n")
    
    # Append to training data
    output_file = "src/data/hineng.train"
    print(f"Adding augmented data to {output_file}...")
    write_augmented_data(augmented, output_file)
    
    print("\n" + "="*70)
    print("DATA AUGMENTATION COMPLETE")
    print("="*70)
    print(f"Added {len(augmented)} synthetic ORG sentences")
    print(f"Total training data updated: {output_file}")
    print("\nNext: Use external Hindi NER datasets for transfer learning")
    print("="*70 + "\n")
