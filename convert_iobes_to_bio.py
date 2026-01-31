# coding=utf-8
"""
Convert IOBES tagging scheme to BIO tagging scheme
IOBES: I-PER, B-PER, E-PER, S-PER -> BIO: B-PER, I-PER, O
This simplifies the tagging scheme and improves learning
"""
import os

def convert_iobes_to_bio(tag):
    """Convert single IOBES tag to BIO"""
    if tag == 'O':
        return 'O'
    
    # Extract prefix and entity type
    parts = tag.split('-', 1)
    if len(parts) != 2:
        return tag
    
    prefix, entity = parts
    
    # Map IOBES prefixes to BIO
    if prefix in ['B', 'S']:  # B-ENT or S-ENT -> B-ENT
        return f'B-{entity}'
    elif prefix in ['I', 'E']:  # I-ENT or E-ENT -> I-ENT
        return f'I-{entity}'
    else:
        return tag

def convert_file(input_path, output_path):
    """Convert a data file from IOBES to BIO"""
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    converted = []
    for line in lines:
        line = line.strip()
        if not line:  # Empty line (sentence boundary)
            converted.append('')
        else:
            # Format: word TAB tag
            parts = line.split('\t')
            if len(parts) == 2:
                word, tag = parts
                new_tag = convert_iobes_to_bio(tag)
                converted.append(f'{word}\t{new_tag}')
            else:
                converted.append(line)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(converted))
    
    print(f"Converted: {input_path} -> {output_path}")

# Convert all data files
data_dir = "src/data"
files_to_convert = [
    ("hineng.train", "hineng.train"),
    ("hineng.valid", "hineng.valid"),
    ("hineng.test", "hineng.test"),
]

print("Converting IOBES tags to BIO...")
print("=" * 60)

for input_file, output_file in files_to_convert:
    input_path = os.path.join(data_dir, input_file)
    output_path = os.path.join(data_dir, output_file)
    
    if os.path.isfile(input_path):
        convert_file(input_path, output_path)
    else:
        print(f"Warning: {input_path} not found")

print("=" * 60)
print("Conversion complete! Files updated with BIO tags.")
