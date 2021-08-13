# A systematic approach to identify the information captured by Knowledge Graph Embeddings

This project contains the code developed for the experiments presented in the paper "A systematic approach to identify 
the information captured by Knowledge Graph Embeddings".
Due to confidentiality constraints, we are not allowed to release data concerning the OntoSIDES Knowledge Graph, 
therefore the present project only includes scripts and data to reproduce the experiments on the subgraph of YAGO3 
presented in the paper.

## Project content
The structure of the project is as follows:
* **./classifiers/** contains the scripts to train and test the classifiers for each one of the probing tasks defined 
* for the YAGO3 subgraph;
* **./graphs/YAGO3/** contains the subgraph used for the experiments extracted as described in the paper;
* **./results/YAGO3/** contains the results presented in the paper;
* **./datasets/YAGO3/** contains the file yago_types.csv to be used to train the classifier for the entity's type 
and the directory gembs/ where the computed GEs for each model should be placed (See "Computing Graph Embeddings").

## Reproducing the experiments
To reproduce the experiments and obtain the results presented in the paper you need first to compute the 
Graph Embeddings from the graph located in the ./graphs/YAGO3/ and place them in ./datasets/YAGO3/gembs/ 
(See "Computing Graph Embeddings") .
To execute the scripts included in this project you will need a Python environment containing the packages list in 
the file requirements.txt.
Once your Python environment is ready, you can reproduce the experiments by executing the file main.py importing the 
classifier you want to test. **main.py** will test all the GE models on the same probing task (based on the classifier 
you imported), and output the results in the corresponding subdirectory of **./results/YAGO3/** .

## Computing Graph Embeddings
Due to the restrictions on the file size imposed by GitHub is not possible for us to upload the computed GEs along 
with the present project. Therefore, to reproduce the results presented in the paper you will need to compute the GEs 
of the graph located in **./graphs/YAGO3/** and place them in **./datasets/YAGO3/gembs/** .

### node2vec
To compute the GEs with node2vec, we used the framework "SNAP for C++: Stanford Network Analysis Platform" 
(http://snap.stanford.edu/index.html). The command used to compute the graph embeddings is the following:
```
./node2vec -i:./graphs/YAGO3/yago.edgelist -o:./datasets/YAGO3/node2vec/yago.emb -l:15 -dr -d:100 -v -p:0.3
```
The file **./graphs/YAGO3/yago.edgelist** contains the graph representation in the required format to be used as 
input for the GE computation.
The output (list of embeddings for each node) will be saved in **./datasets/YAGO3/yago.emb**.
At this point you can run the script **./node2vec2CSV.py** to associate each graph embedding to the URI of the 
corresponding node and save it as a CSV file that will be used by the classification scripts.


### KGEs: ComplEx, DistMult, RESCAL, RotatE and TransE
To compute the GEs with the other models tested in the paper we made use of the AWS DGL-KGE library
(https://aws-dglke.readthedocs.io/en/latest/index.html). The commands used for the GEs computation for each model are 
listed hereafter.

#### ComplEx
```
DGLBACKEND=pytorch dglke_train --model_name ComplEx --data_path ./ --dataset yago --format raw_udd_hrt --data_files yago.txt --delimiter '|' --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 100 --gamma 143.0 -adv \
--lr 0.25 --batch_size_eval 16 --gpu 0 1 --async_update --max_step 10000 --rel_part --force_sync_interval 1000
```

#### DistMult
```
DGLBACKEND=pytorch dglke_train --model_name DistMult --data_path ./ --dataset yago --format raw_udd_hrt --data_files yago.txt --delimiter '|' --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 100 \
--lr 0.25 --batch_size_eval 16 --gpu 0 1 --async_update --max_step 10000 --force_sync_interval 1000
```

#### RESCAL
```
DGLBACKEND=pytorch dglke_train --model_name RESCAL --data_path ./ --dataset yago --format raw_udd_hrt --data_files yago.txt --delimiter '|' --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 100 --gamma 24.0 -adv \
--lr 0.25 --batch_size_eval 16 --gpu 0 1 --async_update --max_step 10000 --rel_part --force_sync_interval 1000
```

#### RotatE
```
DGLBACKEND=pytorch dglke_train --model_name RotatE --data_path ./ --dataset yago --format raw_udd_hrt --data_files yago.txt --delimiter '|' --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 50 --gamma 19.9 -adv -de --neg_deg_sample \
--lr 0.25 --batch_size_eval 16 --gpu 0 1 --async_update --max_step 10000 --rel_part --force_sync_interval 1000
```

#### TransE
```
DGLBACKEND=pytorch dglke_train --model_name TransE_l2 --data_path ./ --dataset yago--format raw_udd_hrt --data_files yago.txt --delimiter '|' --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 100 --gamma 19.9 \
--lr 0.25 --batch_size_eval 16 --gpu 0 1 --async_update --max_step 10000 --force_sync_interval 1000
```

The input file **yago.txt** (located in **./graphs/YAGO3/**) contains the list of the triples in the graph, separated 
by a '|' character.
The output files containing the GEs of entities and relations are saved in the subdirectories **./ckpts/{GE_MODEL}_yago_0/**.
These files need to be moved in the corresponding directories **./datasets/YAGO3/gembs/{GE_MODEL}/** together with the 
two mapping files **entities.tsv** and **relations.tsv** to associate each computed embeddings to the URI of the 
corresponding graph element. 
At this point you can run the script **./DGLKGE2CSV.py** to associate each graph embedding to the URI of the 
corresponding node and save it as a CSV file that will be used by the classification scripts.

