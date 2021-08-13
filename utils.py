import sys, getopt, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_command_line_options(options_list):
    options_string = ":"
    options_string = options_string.join(options_list) + ':'
    try:
        opts, args = getopt.getopt(sys.argv[1:], options_string)
    except getopt.GetoptError:
        sys.exit(2)
    options_values = dict()
    for opt, arg in opts:
        key = opt[1:]
        options_values[key] = arg
    return options_values


def converting_results(data):
    data.loc[data['result'] < 1, 'result'] = 0
    data['result'] = data['result'].astype(int)
    return data


def get_df_intersection(df_1, df_2, column):
    data = df_1.loc[df_1[column].isin(df_2[column])]
    return data


def get_df_difference(df_1, df_2, column):
    data = df_1.loc[~df_1[column].isin(df_2[column])]
    return data


def merge_graphs(path, pattern):
    from rdflib import Graph
    matching_files = [f for f in os.listdir(path) if pattern in f]
    g = Graph()
    for file in matching_files:
        print(f"Treating file {file}")
        tmp_g = Graph()
        tmp_g.parse(path + file, format='turtle')
        g += tmp_g
    return g


def convert_IRIs_to_IDs(data):
    # Get IDs of answers, questions and students
    data['answer'] = data['answer'].str.replace('http://www.side-sante.fr/sides#answer', '')
    data['question'] = data['question'].str.replace('http://www.side-sante.fr/sides#q', '')
    data['student'] = data['student'].str.replace('http://www.side-sante.fr/sides#stu', '')

    # Casting to int
    data['answer'] = data['answer'].astype(int)
    data['question'] = data['question'].astype(int)
    data['student'] = data['student'].astype(int)
    return data


def get_IRI_strings(data, column):
    return data[column].str.replace('<', '').str.replace('>', '')


def format_IRI(IRI, NAMESPACE_LIST):
    name = str(IRI)
    for p in NAMESPACE_LIST:
        name = name.replace(NAMESPACE_LIST[p], p+'_')
    return name


def save_id_index_map(index_list, ids, file):
    import pandas as pd
    pd.DataFrame({'index': index_list, 'ids': ids}).to_csv(file, index=False)


def get_id_index_mapping(identifiers, file):
    import pandas as pd
    records = pd.read_csv(file)['ids'].tolist()
    ids = []
    not_found = []
    for j, i in enumerate(identifiers):
        if i in records:
            ids.append(records.index(i))
        else:
            not_found.append(j)
    return ids, not_found


def parse_DeepFM_logs(filename, field_sep=' - ', value_sep=': ', ignore_header=1, ignore_pattern='Params Configuration'):
    import pandas as pd
    output_file = filename.replace('.log', '_{}.csv')
    data = {
        'n': [],
    }
    f = open(filename)
    content = f.readlines()
    lines = [x.strip() for x in content]
    j = 0
    epoch = 1
    for i in range(0, len(lines)):
        if i < ignore_header:
            continue
        line = lines[i]
        if ignore_pattern in line:
            continue
        if 'Train on ' in line:
            if len(data['n']) > 0:
                pd.DataFrame(data).to_csv(output_file.format(j), index=False)
                j = j + 1
                epoch = 1
                data = {
                    'n': [],
                }
            continue
        if 'Epoch' in line:
            continue
        chunks = line.split(field_sep)
        data['n'].append(epoch)
        epoch = epoch + 1
        for chunk in chunks:
            if value_sep in chunk:
                elements = chunk.split(value_sep)
                name = elements[0]
                value = elements[1]
                if not (name in data.keys()):
                    data[name] = []
                data[name].append(value)
    pd.DataFrame(data).to_csv(output_file.format(j), index=False)


def balance_data(df, attr):
    frequency = df[attr].value_counts()
    mean_el = round(np.percentile(frequency.values, 50))
    # mean_el = round(np.mean(frequency.values))
    balanced = pd.DataFrame()
    for index, value in zip(frequency.index, frequency.values):
        to_sample = df[df[attr] == index]
        if value > mean_el:
            sampled = to_sample.sample(n=mean_el)
        else:
            sampled = to_sample
        balanced = pd.concat([balanced, sampled])
    return balanced


def plot_distribution(distrib, TASK, DATASET):
    fig = plt.figure(figsize=(8, 8))
    plt.bar(range(0, len(distrib)), distrib.values, color='red', tick_label=[str(i) for i in distrib.index.to_list()])
    plt.xlabel(TASK)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90, fontsize=10)
    # plt.xlim([0, max_val])
    # plt.ylim([0, max_val])
    plt.title(f'{DATASET} {TASK}')
    plt.tight_layout()
    plt.savefig(f'../results/{DATASET}/{TASK}/distribution.png')
    plt.show()
