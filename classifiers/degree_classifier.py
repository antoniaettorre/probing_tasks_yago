import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import logging
import utils
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix, f1_score


def get_datasets(embs, triples, th=100):
    if TASK == 'outdegree':
        use = 'subject'
        other = 'object'
    else:
        use = 'object'
        other = 'subject'
    # Count in/out degree based on number of times a object/subject appears in the set of triples
    elements = triples[use].value_counts().to_frame()
    elements.reset_index(inplace=True)
    elements.rename(columns={use: 'degree', 'index': 'iri'}, inplace=True)
    # Identify entities that do not appear as object/subject and add them with degree 0
    zero_degree = triples.loc[~triples[other].isin(elements['iri'])][other].unique()
    elements = pd.concat([elements, pd.DataFrame({'iri': zero_degree, 'degree': 0})])
    # Keep only the classes (degrees) with a reasonable number of samples
    elements = get_most_frequent_degrees(elements, th)
    if BALANCE:
        elements = utils.balance_data(elements, 'degree')
    # Merge with Graph Embeddings
    df = pd.merge(elements, embs, how='left', on='iri')
    y = []
    # Shuffle the dataset
    df = df.sample(len(df))
    lb = LabelEncoder()
    y = lb.fit_transform(df['degree'])
    logging.info('Labels -> Encoding:')
    enc = '\n'
    for i in range(0, max(y) + 1):
        enc = enc+f'{str(i)} -> {lb.inverse_transform([i])}\n'
    logging.info(str(enc))
    y = df['degree'].to_numpy()
    return df[[str(i) for i in range(0, GE_DIM[GE_MODEL])]].to_numpy(), y


def get_most_frequent_degrees(df, th):
    frequent = df['degree'].value_counts()
    frequent = frequent.loc[frequent > th].index
    return df.loc[df['degree'].isin(frequent)]


BASE_DIR = '../'
CONF_MATRIX = True
BALANCE = True
DATASET = 'YAGO3'
TASK = 'indegree'
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

def start():
    GRAPHS = {
        'YAGO3': f'{BASE_DIR}graphs/YAGO3/yago.txt',
    }
    LOGFILE = f'{BASE_DIR}results/{DATASET}/{TASK}/{GE_MODEL}.log'
    logging.basicConfig(filename=LOGFILE, filemode='w', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')


    # Read the graph in csv format
    dataset = pd.read_csv(GRAPHS[DATASET], names=['subject', 'relation', 'object'], sep=SEP[DATASET])
    dataset.drop(columns=['relation'], inplace=True)

    embs = pd.read_csv(f'{BASE_DIR}datasets/{DATASET}/gembs/{GE_MODEL}/{GE_MODEL}_all.csv')
    X, y = get_datasets(embs, dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logging.info(f'Training set size: {X_train.shape}')
    logging.info(f'Test set size: {X_test.shape}')

    model = LogisticRegressionCV(cv=5, class_weight='balanced', max_iter=200).fit(X_train, y_train)

    res = model.predict(X_test)

    logging.info(f'Classification report in ./results/{DATASET}/{TASK}/{GE_MODEL}.csv\n')
    report = pd.DataFrame(classification_report(y_test, res, zero_division =0, output_dict=True)).transpose()
    report.to_csv(f'{BASE_DIR}results/{DATASET}/{TASK}/{GE_MODEL}.csv', index=True)

    fig = plt.figure(figsize=(8,8))
    max_val = np.stack([y_test, res]).max()
    plt.scatter(y_test, res)
    plt.plot([0, max_val], [0, max_val], color='black', linewidth=0.5)

    plt.xlabel('test')
    plt.ylabel('predicted')
    plt.xlim([0, max_val])
    plt.ylim([0, max_val])
    plt.title(f'{GE_MODEL} F1=%.2f' % f1_score(res, y_test, average='weighted'))
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
                                     normalize='true', include_values=False)
        disp.ax_.set_title('Confusion matrix')
        plt.xticks(rotation=90, fontsize=10)
        plt.savefig(f'{BASE_DIR}results/{DATASET}/{TASK}/CM_{GE_MODEL}.png')
        plt.show()