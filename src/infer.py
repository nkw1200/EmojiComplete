import sys
import os
from datasets import Dataset
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TextClassificationPipeline
import pandas as pd
import glob
from zipfile import ZipFile
import emoji as em

from bert import DATA_COLUMN, DATA_PATH, DATASET_COLUMNS, LABEL_COLUMN, MODEL_NAME


def extractData(zip_path: str, num_samples: int, emoji_map: np.ndarray):
    zipFile = os.path.join(DATA_PATH, zip_path)
    with ZipFile(zipFile, 'r') as zip:
        zip.extractall(DATA_PATH)
    output_path = os.path.join(DATA_PATH, os.path.splitext(zip_path)[0])
    all_files = glob.glob(output_path + "/*.csv")

    li = []

    for filename in all_files:
        temp = pd.read_csv(filename, index_col=None,
                           header=None, names=["Tweet", "Emoji"])
        li.append(temp)

    df = pd.concat(li, axis=0, ignore_index=True)
    df = df.sample(num_samples)

    return df


def main(model_name, mapping, dataset_name='test', num_samples=1, num_scores=5):
    dataset = extractData(f"{dataset_name}.zip", num_samples, mapping)
    num_labels = max(mapping.item().values())+1

    model = DistilBertForSequenceClassification.from_pretrained(
        f'model/{model_name}', num_labels=num_labels)  # len(mapping.item()))
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    pipe = TextClassificationPipeline(
        model=model, tokenizer=tokenizer, return_all_scores=True)

    # Randomly sample
    predictions = pipe(list(dataset[DATA_COLUMN]._values))
    for i, prediction in enumerate(predictions):
        print('-------------------------------------------------------------\n')
        print(f"Original tweet: {dataset[DATA_COLUMN]._values[i]}")
        top_idx = np.argsort(-np.asarray([label['score']
                             for label in prediction]))[:num_scores]
        top_emoji_groups = set()
        top_emojis = []
        for emoji, idx in mapping.item().items():
            if idx in top_idx and idx not in top_emoji_groups:
                top_emojis.append(emoji)
                top_emoji_groups.add(idx)
        # top_emojis = [emoji for emoji, idx in mapping.item().items() if idx in top_idx]
        print(f"Top {num_scores} emojis:\n")
        # print(len(top_idx))
        # print(len(top_emojis))
        # print(top_emojis)
        for j, emoji in enumerate(top_emojis):
            print(em.emojize(
                f":{emoji}: ({round(prediction[top_idx[j]]['score'] * 100, 2)}% confidence)"))

        if "clustered" not in sys.argv[2]:
            print(em.emojize(
                f"\nCorrect emoji: :{dataset[LABEL_COLUMN]._values[i]}:"))
            print()
            continue

        correct_emoji = str(dataset[LABEL_COLUMN]._values[i])

        for emoji, idx in mapping.item().items():
            if emoji == correct_emoji:
                correct_emoji_idx = idx
        correct_emojis = [emoji for emoji, idx in mapping.item(
        ).items() if idx == correct_emoji_idx]
        print("\nCorrect emojis: ", end='')
        print(em.emojize(", ".join(f":{emoji}:" for emoji in correct_emojis)))
    print('-------------------------------------------------------------\n')


if __name__ == "__main__":
    argc = len(sys.argv)
    model_name = sys.argv[1]
    mapping_fname = sys.argv[2]
    dataset_name = sys.argv[3] if argc > 3 else 'test'
    num_samples = int(sys.argv[4]) if argc > 4 else 1
    num_scores = int(sys.argv[5]) if argc > 5 else 5

    mapping = np.load(f'data/{mapping_fname}', allow_pickle=True)

    main(model_name, mapping, dataset_name, num_samples, num_scores)
