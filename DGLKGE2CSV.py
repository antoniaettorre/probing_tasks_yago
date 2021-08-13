import pandas as pd
import numpy as np
import sys
sys.path.append('../')
import logging

GE_MODEL = 'ComplEx'
INPUT_PATH = f'./datasets/YAGO3/gembs/{GE_MODEL}/'
OUTPUT_PATH = f'./datasets/YAGO3/gembs/{GE_MODEL}/'
DATASET = f'yago_{GE_MODEL}'

ID_FILE = INPUT_PATH + 'entities.tsv'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

logging.info("Reading IDs file...")
ids = pd.read_csv(ID_FILE, names=['id', 'iri'], index_col=0, sep='|')
logging.info("Loading embeddings...")
data = np.load(INPUT_PATH + DATASET + '_entity.npy')
df = pd.DataFrame(data, index=ids.index)
logging.info("Merging...")
df = pd.merge(ids, df, left_index=True, right_index=True)

df.to_csv(OUTPUT_PATH + GE_MODEL + '_all.csv', index=False)
