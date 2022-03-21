import simpledorff
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import argparse
import ipdb
import numpy as np
from collections import defaultdict
import json
import math
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def print_pearson(reliability_data):
    n_anns = reliability_data.shape[1]
    for i in range(n_anns):
        for j in range(i+1,n_anns):
            x = reliability_data[:,i]
            y = reliability_data[:,j]
            data = np.array([[a,b] for (a,b) in zip(x,y) if not math.isnan(a) and not math.isnan(b)])
            print(pearsonr(data[:,0], data[:,1]))


def remove_invalid_values(annotations):
    '''
    Perform validation on the annotations, turn invalid annotations into NaN
    :param annotations:
    :return:
    '''

    for idx, row in annotations.iterrows():
        if isinstance(row['Fluency'], str) and int(row['Fluency'][0]) < 2:
            annotations.at[idx, 'De-Contextualized'] = math.nan
            annotations.at[idx, 'Atomicity'] = math.nan
            annotations.at[idx, 'Faithfulness'] = math.nan
        if isinstance(row['De-Contextualized'], str) and int(row['De-Contextualized'][0]) == 0:
            annotations.at[idx, 'Atomicity'] = math.nan
            annotations.at[idx, 'Faithfulness'] = math.nan
    return annotations


def calculate_average_scores_per_method(annotations):
    columns = ['Fluency', 'De-Contextualized', 'Atomicity', 'Faithfulness']
    scores_by_method = defaultdict(lambda: {col: [] for col in columns})
    missing_by_method = defaultdict(lambda: {col: 0 for col in columns})
    acceptable_by_method = defaultdict(int)
    total_by_method = defaultdict(int)
    for idx,row in annotations.iterrows():
        col_values = []
        for col in columns:
            if not isinstance(row[col], str) and math.isnan(row[col]):
                missing_by_method[row['Method']][col] += 1
            else:
                scores_by_method[row['Method']][col].append(
                    float(row[col][0]) if isinstance(row[col], str) else row[col])
            col_values.append(float(row[col][0]) if isinstance(row[col], str) else row[col])
        total_by_method[row['Method']] += 1
        if col_values[0] > 1 and col_values[2] == 1 and col_values[3] > 3:
            acceptable_by_method[row['Method']] += 1



    method_graph_data = defaultdict(lambda: {col: [] for col in columns})
    for method in sorted(scores_by_method):
        base_method = method[:method.rfind('_')]
        for col in columns:
            score = np.mean(scores_by_method[method][col])
            method_graph_data[base_method][col].append(score)
            #print(f"{method},{col}: {score}\t{len(scores_by_method[method][col])}")
        print(f"{method},accepted: {acceptable_by_method[method]}\ttotal: {total_by_method[method]}\tprecision: {acceptable_by_method[method] / total_by_method[method]}")
        #print(f"{method},precision: {acceptable_by_method[method] / total_by_method[method]}")

    for method in method_graph_data:
        for col in method_graph_data[method]:
            fig = plt.figure(figsize=(5,5))
            ax = sns.lineplot(x=list(range(len(method_graph_data[method][col]))), y=method_graph_data[method][col])
            ax.set_title(col)
            fig.tight_layout()
            fig.savefig(f'./annotation_data/main_experiment_1/{method}_{col}.pdf')
            fig.clf()


def get_annotation_median_score(annotations_multi):
    columns = ['Fluency', 'De-Contextualized', 'Atomicity', 'Faithfulness']
    out_annotations = annotations[0].copy()
    for col in columns:
        data = np.array(
            [[float(val[0]) if isinstance(val, str) else val for val in ann[col]] for ann in annotations_multi])
        out_annotations[col] = [np.median(row) for row in data]
    return out_annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_files", nargs='+', help="List of annotation files", default=[])
    args = parser.parse_args()
    annotations = [remove_invalid_values(pd.read_csv(f).dropna(how='all').fillna(np.nan)) for f in args.annotation_files]

    if len(args.annotation_files) > 1:
        # Combine claims for dataset, separate multi-annotated claims from singly annotated claims
        split_point = 0
        while all(a.values[split_point,0] == annotations[0].values[split_point,0] for a in annotations):
            split_point += 1
        annotations_multi = [a[:split_point] for a in annotations]
        annotations_compare = annotations_multi[0].copy()

        print([a.shape for a in annotations_multi])
        columns = ['Fluency', 'De-Contextualized', 'Atomicity', 'Faithfulness']
        value_domains = [[0,1], [0,1], [0,1], [0,1]]
        for col,dom in zip(columns,value_domains):
            reliablility_data = np.array([[float(val[0]) if isinstance(val, str) else val for val in ann[col]] for ann in annotations_multi])
            comp = [" | ".join(str(int(r)) if not math.isnan(r) else str(r) for r in row) for row in reliablility_data.T]
            annotations_compare[col] = comp
            if col == 'Fluency':
                print_pearson(reliablility_data.T)
                reliablility_data[reliablility_data == 1] = 0
                reliablility_data[reliablility_data > 1] = 1
            #     print(f"Fluency % all equal: {sum(np.all(row == row[0]) for row in reliablility_data.T) / reliablility_data.shape[1]}")
            if col == 'Faithfulness':
                print_pearson(reliablility_data.T)
                for i in range(reliablility_data.shape[0]):
                    for j in range(reliablility_data.shape[1]):
                        if not math.isnan(reliablility_data[i,j]):
                            if reliablility_data[i,j] > 3:
                                reliablility_data[i,j] = 1
                            else:
                                reliablility_data[i,j] = 0

            # Get straight disagreement %
            collapsed_agreement = []
            for row in reliablility_data.T:
                vals = [r for r in row if r != np.nan]
                if len(vals) > 0:
                    collapsed_agreement.append(all(v == vals[0] for v in vals))

            # convert into dataframe for simpledorff
            array_of_reliability_data = []
            for annotator_index, annotation_values in enumerate(reliablility_data):
                for entry_index, value in enumerate(annotation_values):
                    array_of_reliability_data.append([entry_index, annotator_index, value])
            df_reliability = pd.DataFrame(array_of_reliability_data, columns=['claim_id', 'annotator_id', 'annotation'])
            alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
                df_reliability,
                experiment_col='claim_id',
                annotator_col='annotator_id',
                class_col='annotation'
            )

            print(f"{col}: {alpha}\t{sum(collapsed_agreement) / len(collapsed_agreement)}")
        all_annotations = pd.concat([get_annotation_median_score(annotations_multi)] + [ann[split_point:] for ann in annotations])
    else:
        all_annotations = annotations[0]
    calculate_average_scores_per_method(all_annotations)