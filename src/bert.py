
"""BERT_sentiment_sample.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1E0cy1v9xS7e61hxs4tFgVcA1C_qcy51R
"""

from typing import List
from zipfile import ZipFile
import torch
import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
import glob

from plot import plot_loss

DATA_COLUMN = 'Tweet'
LABEL_COLUMN = 'Emoji'
MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = 'data'
DATASET_COLUMNS = ['input_ids', 'attention_mask', 'labels']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 3

def extractData(zip_path: str, debug: bool):
    """**Download Data**"""
    zipFile = os.path.join(DATA_PATH, zip_path)
    with ZipFile(zipFile, 'r') as zip:
        zip.extractall(DATA_PATH)
    output_path = os.path.join(DATA_PATH, os.path.splitext(zip_path)[0])
    all_files = glob.glob(output_path + "/*.csv")

    li = []

    for filename in all_files:
        temp = pd.read_csv(filename, index_col=None, header=None, names=["Tweet","Emoji"])
        li.append(temp)

    df = pd.concat(li, axis=0, ignore_index=True)
    if debug:
        df = df[:100]

    """# Make Datasets"""

    mappingPath = os.path.join(DATA_PATH, 'mapping.npy')
    mapping_emoji = np.load(mappingPath, allow_pickle=True)

    df[LABEL_COLUMN].replace(mapping_emoji,
                            [i for i in range(len(mapping_emoji))],
                            inplace=True)

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda e: tokenizer(e[DATA_COLUMN], 
                                            truncation=True, 
                                            padding='max_length'), batched=True)

    dataset = dataset.rename_column(LABEL_COLUMN, "labels")
    dataset.set_format(type='torch', columns=DATASET_COLUMNS)

    return (dataset, mapping_emoji)

def main(dataset_name: str, debug: bool):
    print('Extracting data... \n')
    dataset, mapping_emoji = extractData(f"{dataset_name}.zip", debug)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, 
                                                                num_labels=len(mapping_emoji))
    model = model.to(DEVICE)

    train_testvalid = dataset.train_test_split(test_size=0.1)
    train = train_testvalid["train"]
    test = train_testvalid["test"]

    """# Create the Model

    Further hyperparameter tuning is needed here
    """

    optimizer = torch.optim.Adam(params = model.parameters(), lr=1e-4, eps=1e-08)
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 1 if debug else 16
    dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size)

    print('Training... \n')
    loss_history = np.empty(len(dataloader))
    # From HuggingFace quickstart
    for epoch in range(EPOCHS):
        print(f'\n\nEpoch {epoch+1}/3:')
        loss = 10
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss_history[i] = loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print(f"\nloss: {loss}")

    print('Evaluating... \n')
    model.eval()

    test_dict = {k: test[k].to(DEVICE) for k in DATASET_COLUMNS}
    out = model(**test_dict)
    print(classification_report(test['labels'].cpu(), torch.max(out['logits'], axis=1)[1].cpu()))

    print('Saving model... \n')
    model.save_pretrained(f'model/{dataset_name}-trained-bert')

    print('Creating figure... \n')
    
    plot_loss('Bert Training', loss_history, EPOCHS)
    # Save loss history in case we want to regenerate graph
    np.save(f'data/bert_{dataset_name}_loss.npz', loss_history)

if __name__ == "__main__":
    dataset = sys.argv[1]
    debug = sys.argv[2] == 'true'
    main(dataset, debug)
