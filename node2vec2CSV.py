import csv
import pandas as pd
import networkx as nx
import sys
sys.path.append('../')
import logging
import utils


EMB_FILE = './datasets/YAGO3/gembs/node2vec/yago.emb'
NX_FILE = './graphs/YAGO3/yago.gpickle'

EXPORT_FILE = './datasets/YAGO3/gembs/node2vec/node2vec_all.csv'

NAMESPACE_LIST = {
    'owl': 'http://www.w3.org/2002/07/owl#',
    'yago': 'http://yago-knowledge.org/resource/',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'schema': 'https://schema.org/',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'xml': 'http://www.w3.org/2001/XMLSchema#',
    'xsd': 'http://www.w3.org/XML/1998/namespace'
}

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')


logging.info("Reading gpickle...")
nxDiGraphInt = nx.read_gpickle(NX_FILE)

nodes = list(nxDiGraphInt.nodes(data=True))
id_to_label = {k:utils.format_IRI(str(v['old_label']), NAMESPACE_LIST) for k,v in nodes}

logging.info("Reading embeddings...")
with open(EMB_FILE, "r") as embeddings:
    next(embeddings)
    reader = csv.reader(embeddings, delimiter=" ")

    embs = []
    i = 1
    init = 0
    for row in reader:
        if i <= init:
            i = i + 1
            continue
        node_id = row[0]
        embs.append({
            "node_id": int(node_id),
            "embedding": [float(item) for item in row[1:]]
        })
        i = i + 1

logging.info(f"Generating file")
df = pd.DataFrame(embs)
df['iri'] = df['node_id'].map(id_to_label)
df = df.drop('node_id', axis=1)

# expand df_1.embedding into its own dataframe
df_embs = df['embedding'].apply(pd.Series)

df_expanded = pd.concat([df[:], df_embs[:]], axis=1)
df_expanded = df_expanded.drop('embedding', axis=1)
logging.info(f"Exporting file")
df_expanded.to_csv(EXPORT_FILE, index = False)


