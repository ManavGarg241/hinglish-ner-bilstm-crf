# ✅ Hinglish NER (BiLSTM-CRF + Transfer Learning)

A deep learning based **Named Entity Recognition (NER)** system for **Hinglish code-mixed text**
built using **BiLSTM-CRF** in PyTorch, enhanced with:

- Transfer learning from **WikiAnn Hindi**
- Multilingual **FastText embeddings**
- Span-level evaluation using **SeqEval**

✅ Best Validation Span F1: **0.84**

---

# 🚀 Project Highlights

- ✅ BiLSTM-CRF sequence labeling architecture  
- ✅ Hinglish (Hindi-English code-mixed) dataset support  
- ✅ Transfer Learning: WikiAnn Hindi → Hinglish fine-tuning  
- ✅ Multilingual FastText pretrained embeddings  
- ✅ Span-level evaluation using SeqEval  
- ✅ GPU supported (RTX tested)  
- ✅ Full preprocessing pipeline (IOBES → BIO conversion)

---

# 📌 Problem Statement

Most NLP models fail on **code-mixed languages** like Hinglish, widely used in:

- Indian social media posts  
- WhatsApp / chat messages  
- Informal search queries  

This project solves the task of automatically extracting named entities from Hinglish text.

---

# 🎯 Use Cases

Hinglish NER is useful for:

- Social media entity extraction  
- Hinglish chatbots and assistants  
- Information retrieval from mixed-language text  
- Search engines and recommendation systems  
- Low-resource multilingual NLP research  

---

# 📂 Dataset

This project uses the **LINCE Code-Mixed Dataset (Hinglish NER)**.

⚠️ Dataset is not uploaded in this repo due to size/licensing.

Download here:  
https://ritual.uh.edu/lince/

Place files inside:

dataset/
- ner_hineng_train.csv  
- ner_hineng_valid.csv  
- ner_hineng_test.csv  

---

# ⚙️ Installation

## Step 1: Create Virtual Environment

python -m venv ner_env

Activate:

Windows:
ner_env\Scripts\activate

Linux/Mac:
source ner_env/bin/activate

---

## Step 2: Install Requirements

pip install -r requirements.txt

---

# 🏋️ Training Pipeline

## Step 1: Pretrain on WikiAnn Hindi

python src/pretrain_and_finetune_wikiann.py

## Step 2: Fine-tune on Hinglish Dataset

python src/train.py --name hinglish_ner --use_gpu 1

---

# 📊 Results

Best Validation Performance (Span-Level):

| Entity Type | F1 Score |
|------------|----------|
| PERSON     | ~0.86    |
| ORG        | ~0.78    |
| PLACE      | ~0.75    |

✅ Overall Best Span-Level F1: **0.84**

---

# 🔍 Inference Example

python src/inference.py "Modi visited Delhi today"

Output:

Modi  → PERSON  
Delhi → PLACE  

---

# 📁 Repository Structure

src/
- train.py  
- model.py  
- loader.py  
- eval.py  
- inference.py  
- pretrain_and_finetune_wikiann.py  

models/
- mapping.pkl  

dataset/ (not uploaded)

README.md  
requirements.txt  
LICENSE  
.gitignore  

---

# 🛠 Technologies Used

- Python  
- PyTorch  
- BiLSTM + CRF  
- FastText multilingual embeddings  
- SeqEval span evaluation  
- Transfer Learning (WikiAnn → Hinglish)

---

# 📜 License

This project is released under the **MIT License**.

---

# 👨‍💻 Author

Manav Garg

---
requests
visdom
