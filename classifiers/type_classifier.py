import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix, f1_score
import utils


def get_nodes_types(embs, types):
    df = pd.merge(embs, types, how='inner', left_on='iri', right_on='node')
    lb = LabelEncoder()
    labels = lb.fit_transform(df['type'].to_numpy())
    enc = '\n'
    for i in range(0, max(labels) + 1):
        enc = enc + f'{str(i)} -> {lb.inverse_transform([i])}\n'
    logging.info(str(enc))
    return df[[str(i) for i in range(0, GE_DIM[GE_MODEL])]].to_numpy(), labels


def get_most_frequent_types(types, th):
    frequent_types = types['type'].value_counts()
    frequent_types = frequent_types.loc[frequent_types > th].index
    return types.loc[types['type'].isin(frequent_types)]


BASE_DIR = '../'
CROSS_VALIDATION = True
TASK = 'type'
DATASET = 'YAGO3'
GE_MODEL = 'RotatE'
GE_DIM = {
    'node2vec': 100,
    'TransE': 100,
    'ComplEx': 100,
    'RESCAL': 100,
    'DistMult': 100,
    'RotatE': 100,
}
BALANCE = True

def start():
    TYPES_FILE = {
        'YAGO3': f'{BASE_DIR}datasets/{DATASET}/yago_types.csv'
    }

    LOGFILE = f'{BASE_DIR}results/{DATASET}/{TASK}/{GE_MODEL}.log'
    logging.basicConfig(filename=LOGFILE, filemode='w', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

    embs = pd.read_csv(f'{BASE_DIR}datasets/{DATASET}/gembs/{GE_MODEL}/{GE_MODEL}_all.csv')

    types = pd.read_csv(TYPES_FILE[DATASET])
    types = get_most_frequent_types(types, 1000)
    if BALANCE:
        types = utils.balance_data(types, 'type')

    types = types.drop_duplicates(subset=['node'], keep='first')

    X, y = get_nodes_types(embs, types)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logging.info(f'Training set size: {X_train.shape}')
    logging.info(f'Test set size: {X_test.shape}')

    model = LogisticRegressionCV(cv=5, class_weight='balanced').fit(X_train, y_train) if CROSS_VALIDATION \
        else LogisticRegression(class_weight='balanced').fit(X_train, y_train)
    res = model.predict(X_test)

    logging.info(f'Classification report in ./results/{DATASET}/{TASK}/{GE_MODEL}.csv\n')
    report = pd.DataFrame(classification_report(y_test, res, zero_division=0, output_dict=True)).transpose()
    report.to_csv(f'{BASE_DIR}results/{DATASET}/{TASK}/{GE_MODEL}.csv', index= True)

    fig = plt.figure(figsize=(8, 8))
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
    disp = plot_confusion_matrix(model, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 include_values=False)
    disp.ax_.set_title('Confusion matrix')
    logging.info('F1-score micro:' + str(f1_score(res, y_test, average='micro')))
    logging.info('F1-score macro:' + str(f1_score(res, y_test, average='macro')))
    logging.info('F1-score weighted:' + str(f1_score(res, y_test, average='weighted')))
    plt.savefig(f'{BASE_DIR}results/{DATASET}/{TASK}/CM_{GE_MODEL}.png')
    plt.show()
