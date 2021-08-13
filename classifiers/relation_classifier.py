import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix, f1_score

import utils


def get_datasets(embs, sample):
    df = pd.merge(sample, embs, how='left', right_on='iri', left_on='subject')
    df.rename(columns={str(i): 'subj_gemb'+str(i) for i in range(0, GE_DIM[GE_MODEL])}, inplace=True)
    df = pd.merge(df, embs, how='left', right_on='iri', left_on='object')
    df.rename(columns={str(i): 'obj_gemb' + str(i) for i in range(0, GE_DIM[GE_MODEL])}, inplace=True)
    lb = LabelEncoder()
    rels = lb.fit_transform(df['relation'])
    enc = '\n'
    for i in range(0, max(rels) + 1):
        enc = enc + f'{str(i)} -> {lb.inverse_transform([i])}\n'
    logging.info(str(enc))
    return df[['subj_gemb' + str(i) for i in range(0, GE_DIM[GE_MODEL])] + ['obj_gemb' + str(i) for i in range(0, GE_DIM[GE_MODEL])]].to_numpy(), rels


def get_most_frequent_relations(rels, th):
    frequent_rels = rels['relation'].value_counts()
    frequent_rels = frequent_rels.loc[frequent_rels > th].index
    return rels.loc[rels['relation'].isin(frequent_rels)]


BASE_DIR = '../'
BALANCE = True
CONF_MATRIX = True
CROSS_VALIDATION = True
DATASET = 'YAGO3'
TASK = 'relations'
GE_MODEL = 'RotatE'
GE_DIM = {
    'node2vec': 100,
    'TransE': 100,
    'ComplEx': 100,
    'RESCAL': 100,
    'DistMult': 100,
    'RotatE': 100,
}
SEP = {'YAGO3': '|'}
size = 10000


def start():
    GRAPHS = {
        'YAGO3': f'{BASE_DIR}/graphs/YAGO3/yago.txt',
    }
    LOGFILE = f'{BASE_DIR}results/{DATASET}/{TASK}/{GE_MODEL}.log'
    logging.basicConfig(filename=LOGFILE, filemode='w', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

    # Read the graph in csv format
    dataset = pd.read_csv(GRAPHS[DATASET], names=['subject', 'relation', 'object'], sep=SEP[DATASET])
    # keep only relations with at least 1000 occurrences
    dataset = get_most_frequent_relations(dataset, 1000)
    if BALANCE:
        dataset = utils.balance_data(dataset, 'relation')
    else:
        dataset = dataset.sample(size)

    embs = pd.read_csv(f'{BASE_DIR}datasets/{DATASET}/gembs/{GE_MODEL}/{GE_MODEL}_all.csv')
    X, y = get_datasets(embs, dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logging.info(f'Training set size: {X_train.shape}')
    logging.info(f'Test set size: {X_test.shape}')

    model = LogisticRegressionCV(cv=5, class_weight='balanced').fit(X_train, y_train) if CROSS_VALIDATION \
        else LogisticRegression(class_weight='balanced').fit(X_train, y_train)
    res = model.predict(X_test)

    logging.info(f'Classification report in ./results/{DATASET}/{TASK}/{GE_MODEL}.csv\n')
    report = pd.DataFrame(classification_report(y_test, res, zero_division =0, output_dict=True)).transpose()
    report.to_csv(f'{BASE_DIR}results/{DATASET}/{TASK}/{GE_MODEL}.csv', index= True)

    fig = plt.figure(figsize=(8,8))

    max_val = np.stack([y_test, res]).max()
    plt.scatter(y_test, res)
    plt.plot([0, max_val], [0, max_val], color='black', linewidth=0.5)

    plt.xlabel('test')
    plt.ylabel('predicted')
    plt.xlim([0, max_val])
    plt.ylim([0, max_val])
    plt.title(f'{GE_MODEL} Acc=%.2f' % accuracy_score(y_test, res))
    plt.savefig(f'{BASE_DIR}results/{DATASET}/{TASK}/{GE_MODEL}.png')
    plt.show()
    #
    logging.info('ACC:' + str(np.mean(y_test == res)))
    logging.info('F1-score micro:' + str(f1_score(res, y_test, average='micro')))
    logging.info('F1-score macro:' + str(f1_score(res, y_test, average='macro')))
    logging.info('F1-score weighted:' + str(f1_score(res, y_test, average='weighted')))

    if CONF_MATRIX:
        disp = plot_confusion_matrix(model, X_test, y_test,
                                     cmap=plt.cm.Blues,
                                     normalize='true',
                                     include_values=False)
        disp.ax_.set_title('Confusion matrix')
        plt.savefig(f'{BASE_DIR}results/{DATASET}/{TASK}/CM_{GE_MODEL}.png')
        plt.show()