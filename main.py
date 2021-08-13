from classifiers import degree_classifier
from classifiers import direct_link_classifier
from classifiers import relation_classifier as cls
from classifiers import type_classifier


def test_GE():
    GE_ALGOS = [
        'node2vec',
        'TransE',
        'ComplEx',
        'RESCAL',
        'DistMult',
        'RotatE',
    ]
    # To redefine only for degree classifier to choose between indegree and outdegree
    # cls.TASK = 'indegree'
    cls.BASE_DIR = './'
    for ge in GE_ALGOS:
        print(f'Executing {ge}...')
        cls.GE_MODEL = ge
        cls.start()


if __name__ == '__main__':
    test_GE()

exit()

