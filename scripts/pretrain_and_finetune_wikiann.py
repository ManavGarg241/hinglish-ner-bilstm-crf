"""Pretrain on WikiAnn Hindi then fine-tune on local Hinglish dataset.

This script runs `src/train.py` twice:
 1) Pretrain on WikiAnn Hindi (saves model as models/wikiann_pretrain)
 2) Continue training (fine-tune) on `src/data/hineng.train` using the saved model

Usage: run from repo root: python src/pretrain_and_finetune_wikiann.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
TRAIN_SCRIPT = os.path.join(ROOT, "src", "train.py")


def run(cmd):
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, shell=False)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    # 1) Pretrain on WikiAnn
    wiki_train = os.path.join("src", "data", "wikiann_hi.train")
    wiki_dev = os.path.join("src", "data", "wikiann_hi.dev")
    wiki_test = os.path.join("src", "data", "wikiann_hi.test")

    if not (os.path.isfile(wiki_train) and os.path.isfile(wiki_dev) and os.path.isfile(wiki_test)):
        print("WikiAnn data not found. Run src/download_wikiann_hindi.py first.")
        sys.exit(1)

    pretrain_name = "wikiann_pretrain"

    cmd_pretrain = [
        sys.executable,
        TRAIN_SCRIPT,
        "--name",
        pretrain_name,
        "--train",
        wiki_train,
        "--dev",
        wiki_dev,
        "--test",
        wiki_test,
        "--word_dim",
        "300",
    ]

    run(cmd_pretrain)

    # 2) Fine-tune on Hinglish: continue training by using same --name so model is reloaded
    hinglish_train = os.path.join("src", "data", "hineng.train")
    hinglish_dev = os.path.join("src", "data", "hineng.valid")
    hinglish_test = os.path.join("src", "data", "hineng.test")

    if not (os.path.isfile(hinglish_train) and os.path.isfile(hinglish_dev) and os.path.isfile(hinglish_test)):
        print("Hinglish data not found in src/data/. Ensure you ran augment/convert scripts.")
        sys.exit(1)

    cmd_finetune = [
        sys.executable,
        TRAIN_SCRIPT,
        "--name",
        pretrain_name,  # reuse same name so model is reloaded and continued
        "--train",
        hinglish_train,
        "--dev",
        hinglish_dev,
        "--test",
        hinglish_test,
        "--reload",
        "1",
        "--word_dim",
        "300",
    ]

    run(cmd_finetune)


if __name__ == "__main__":
    main()
