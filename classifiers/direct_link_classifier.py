import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix, f1_score


def get_datasets(embs, sample):
    df = pd.merge(sample, embs, how='left', right_on='iri', left_on='subject')
    df.rename(columns={str(i): 'subj_gemb'+str(i) for i in range(0, GE_DIM[GE_MODEL])}, inplace=True)
    df = pd.merge(df, embs, how='left', right_on='iri', left_on='object')
    df.rename(columns={str(i): 'obj_gemb' + str(i) for i in range(0, GE_DIM[GE_MODEL])}, inplace=True)
    return df[['subj_gemb' + str(i) for i in range(0, GE_DIM[GE_MODEL])] + ['obj_gemb' + str(i) for i in range(0, GE_DIM[GE_MODEL])]].to_numpy(), LabelEncoder().fit_transform(df['relation'])


def create_neg_sample(positive_set, n_samples):
    subjs = positive_set['subject']
    objs = positive_set['object']
    all_nodes = pd.concat([subjs, objs]).drop_duplicates()
    sampled_subjs = all_nodes.sample(n_samples).reset_index(drop=True)
    sampled_objs = all_nodes.sample(n_samples).reset_index(drop=True)
    df = pd.DataFrame({'subject': sampled_subjs, 'object': sampled_objs})
    df = df.loc[df['subject'] != df['object']]
    df['rel'] = df['subject'] + df['object']
    positive_set['rel'] = positive_set['subject'] + positive_set['object']
    df = df.loc[~df['rel'].isin(positive_set['rel'])]
    positive_set.drop(columns=['rel'], inplace=True)
    return df.drop(columns='rel')


BASE_DIR = '../'
CONF_MATRIX = True
CROSS_VALIDATION = True
DATASET = 'YAGO3'
TASK = 'links'
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
# size of the sample to use for training and testing
size = 100000

def start():
    GRAPHS = {
        'YAGO3': f'{BASE_DIR}graphs/YAGO3/yago.txt',
    }
    LOGFILE = f'{BASE_DIR}results/{DATASET}/{TASK}/{GE_MODEL}.log'
    logging.basicConfig(filename=LOGFILE, filemode='w', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

    # Read the graph in csv format and subsample to size
    pos_data = pd.read_csv(GRAPHS[DATASET], names=['subject', 'relation', 'object'], sep=SEP[DATASET])
    pos_data.drop(columns=['relation'], inplace=True)
    neg_sample = create_neg_sample(pos_data, size)
    neg_sample['relation'] = 0
    pos_sample = pos_data.sample(size)
    pos_sample['relation'] = 1

    data = pd.concat([pos_sample, neg_sample])
    data = data.sample(len(data))

    embs = pd.read_csv(f'{BASE_DIR}datasets/{DATASET}/gembs/{GE_MODEL}/{GE_MODEL}_all.csv')
    X, y = get_datasets(embs, data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logging.info(f'Training set size: {X_train.shape}')
    logging.info(f'Test set size: {X_test.shape}')

    model = LogisticRegressionCV(cv=5, class_weight='balanced').fit(X_train, y_train) if CROSS_VALIDATION \
        else LogisticRegression(class_weight='balanced').fit(X_train, y_train)
    res = model.predict(X_test)

    logging.info(f'Classification report in ./results/{DATASET}/{TASK}/{GE_MODEL}.csv\n')
    report = pd.DataFrame(classification_report(y_test, res, zero_division =0, output_dict=True)).transpose()
    report.to_csv(f'{BASE_DIR}results/{DATASET}/{TASK}/{GE_MODEL}.csv', index=True)

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
    #
    logging.info('ACC:' + str(np.mean(y_test == res)))
    logging.info('F1-score:' + str(f1_score(res, y_test)))

    if CONF_MATRIX:
        disp = plot_confusion_matrix(model, X_test, y_test,
                                     cmap=plt.cm.Blues,
                                     normalize='true',
                                     include_values=False)
        disp.ax_.set_title('Confusion matrix')
        plt.savefig(f'{BASE_DIR}results/{DATASET}/{TASK}/CM_{GE_MODEL}.png')
        plt.show()

