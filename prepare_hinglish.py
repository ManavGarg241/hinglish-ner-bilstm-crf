import pandas as pd

def clean_array(text):
    """
    Converts strings like:
    "['Kal' 'Rohit' 'Delhi']"
    into:
    ['Kal','Rohit','Delhi']
    """
    # remove brackets
    text = text.replace("[", "").replace("]", "")
    text = text.replace("'", "")
    return text.split()

def convert_to_bio(csv_file, output_file):
    df = pd.read_csv(csv_file)

    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():

            words = clean_array(row["words"])
            tags  = clean_array(row["ner"])

            # Sanity check
            if len(words) != len(tags):
                continue

            for w, t in zip(words, tags):
                f.write(f"{w} {t}\n")

            f.write("\n")

    print(f"✅ Saved: {output_file}")

if __name__ == "__main__":
    convert_to_bio("dataset/ner_hineng_train.csv", "src/data/hineng.train")
    convert_to_bio("dataset/ner_hineng_validation.csv", "src/data/hineng.valid")
    convert_to_bio("dataset/ner_hineng_test.csv", "src/data/hineng.test")
