import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
import numpy as np
from pathlib import Path
import pandas as pd
import glob
from zipfile import ZipFile
import emoji as em
from matplotlib.font_manager import FontProperties

from bert import DATA_COLUMN, DATA_PATH, DATASET_COLUMNS, LABEL_COLUMN, MODEL_NAME

def extractData(zip_path: str):
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

    return df

# I give up
# def plot(data, title):
#     fpath = Path('/System/Library/Fonts/Apple Color Emoji.ttc')
#     data = {em.emojize(f":{k}:"): v for k,v in data.items()}
#     # mpl.rcParams['pdf.fonttype'] = 42
    
#     fig, ax = plt.subplots()
#     fig.set_figwidth(8)
#     ax.bar(*zip(*data.items()))
#     ax.set_xticklabels(data.keys(), font=fpath)
#     ax.set_title(f'{title} relative distribution')

#     plt.savefig(f'figures/{title}_dist.png')


def main():
    
    dataset = extractData("train.zip")
    feq_table = dict(dataset[LABEL_COLUMN].value_counts()/len(dataset))
    # plot(feq_table, 'One-to-one')
    print(feq_table)



if __name__ == "__main__":
    main()