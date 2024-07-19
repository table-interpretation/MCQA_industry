__author__ = 'leah.michel'

import time

import torch
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import argparse
import logging
import pandas as pd
import os
from ast import literal_eval
import string
import random

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


#################OPTIONAL PREPROCESSING FUNCTIONS####################

def limit_class(df, ent_type, entlab_count, count_limit):
    """takes a dataframe, returns a dataframe with a limited number of rows for specific ent_type.
    @param df: dataframe
    @param ent_type: entity type to limit
    @param entlab_count: dictionary of entity labels and counts
    @param count_limit: maximum number of rows for specific ent_type
    """

    for ent, count in entlab_count.items():
        if count > count_limit:
            mask_ent = df[ent_type] == ent
            df_mask = df[mask_ent][:(count - count_limit)]
            df = df[~df.isin(df_mask)]
            df = df.dropna()

    return df


def clean_datum(datum):
    datum = ''.join(x for x in list(datum) if x.isdigit() or x in ['x', '/', '.', ',', '-', '@'])

    # replace comma with period
    return datum.replace(',', '.') if ',' in datum else datum


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")
    return text


####################################################################

def match_quants(df):
    """ takes a dataframe, adds 'selected_quants' column with quants that have the same unit of measure as the answer quant
    @param df: dataframe with 'Quant' and 'UOM' columns"""

    logger.info("matching quants...")

    df['selected_quants'] = ''
    uom_quants = defaultdict(set)

    for i in range(len(df)):
        quant = df['Quant'].values[i]
        for j in range(1, len(df)):
            next_quant = df['Quant'].values[j]
            uom = df['UOM'].values[j]
            if quant != next_quant:  # answer quant not considered
                uom_quants[uom].add(next_quant.strip().lower())

        df['Datum'].values[i] = clean_datum(df['Datum'].values[i])
        df['Datum'].values[i] = remove_punctuations(df['Datum'].values[i])

        unit = df['UOM'].values[i]  # get the unit of measure of answer quant
        df['selected_quants'].values[i] = list(uom_quants.get(unit, ''))  # get quants with the same unit of measure

        logger.info("Quants with same unit of measure: {}".format(df['selected_quants'].values[i]))

    logger.info("Done matching quants")

    return df


def article(w):
    return 'an ' if w[0].lower() in ['a', 'e', 'i', 'o', 'u'] else 'a '


def add_prompts(df):
    """takes a dataframe, adds 'Question' column with generated prompts for each row"""

    logger.info("adding prompts...")

    df['Question'] = ''

    for i in range(len(df)):
        df['Question'].values[i] = 'Which of the following refers to the ' + str(df['Datum'].values[i]) + ' ' + \
                                   df['UOM'].values[i] \
                                   + ' of ' + article(df['Eq_Label'].values[i]) + df['Eq_Label'].values[i].lower() + '?'

    logger.info("Done adding prompts")
    return df


def filter_choices(selected, choices, k, symmetric=True):
    """returns a list of k choices most semantically similar to the given choice
    @param selected: the selected choice
    @param choices: list of choices
    @param k: number of choices to return
    @param st_model: sentence transformer model to use. msmarco is for asymmetric semantic search
    """

    logger.info("filtering choices...")

    if symmetric:
        st_model = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    else:
        st_model = 'sentence-transformers/msmarco-distilbert-base-v4'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model
    model = SentenceTransformer(st_model, device=device)

    # get embeddings of question and choices
    q_embeddings = model.encode(selected, convert_to_tensor=True)
    ch_embeddings = model.encode(choices, convert_to_tensor=True)

    q_embeddings = q_embeddings.to(device)
    ch_embeddings = ch_embeddings.to(device)

    # get most similar. symmetric uses dot product, asymmetric uses cosine similarity
    # https://www.sbert.net/examples/applications/semantic-search/README.html

    hits = util.semantic_search(q_embeddings, ch_embeddings, score_function=util.cos_sim, top_k=k)

    # get top choices
    filtered_choices = [choices[hits[0][i]['corpus_id']] for i in range(len(hits[0]))]

    logger.info("Done filtering choices")

    return filtered_choices


def add_choices(df, ent, choices, num_choices):
    """takes a dataframe, adds 'filtered_choices' and 'final_choices' columns
    @param df: dataframe
    @param ent: entity type, column name for answer
    @param choices: list of choices
    @param num_choices: number of choices to return
    """
    logger.info("adding choices...")

    df['filtered_choices'] = ''
    df['final_choices'] = ''

    df = add_prompts(df)

    for i in range(len(df)):

        answer = df[ent].values[i]
        #select_quants = list(set(literal_eval(df['selected_quants'].values[i])))
        select_quants = list(set(df['selected_quants'].values[i]))
        print('select quants: ', select_quants)

        # remove selected quants from choices
        choices_new = [c for c in choices if c not in select_quants]

        print('choices: ', choices_new)
        if len(select_quants) == 0:
            # use asymmetric search if there are no matching quants
            filtered = filter_choices(df['Question'].values[i], choices_new, num_choices, symmetric=False)
        elif len(select_quants) >= num_choices:
            filtered = select_quants[:num_choices]
        else:
            filtered = select_quants.copy()
            while len(filtered) < num_choices:
                filt = filter_choices(filtered[0], choices_new, num_choices - len(filtered), symmetric=True)
                filtered.extend(filt)

        df['filtered_choices'].values[i] = filtered
        new_choices = filtered.copy()
        print('filtered: ', filtered)

        if answer not in new_choices:
            new_choices[-1] = answer

        random.shuffle(new_choices)

        df['final_choices'].values[i] = new_choices
        # print('final choices: ', new_choices)
        # print('\n')

    logger.info("Done adding choices")

    return df


def add_labels(df, ent):
    """takes a dataframe, adds 'label' column with index of correct answer in final_choices
    @param df: dataframe
    @param ent: entity type, column name for answer"""

    print('adding labels...')

    df['label'] = ''

    for i in range(len(df)):
        choices = df['final_choices'].values[i]
        df['label'].values[i] = choices.index(df[ent].values[i])

    print('done adding labels')

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True, help="path to input directory")
    parser.add_argument("--entity", type=str, required=True, help="entity type, column name for answer")
    parser.add_argument("--output_dir", type=str, required=True, help="path to output directory")
    parser.add_argument("--posc", type=str, required=True,
                        help="path to posccaesar schema file. only used for synthetic data")
    parser.add_argument("--n", type=int, required=True, help="number of MCQA choices")
    parser.add_argument("--data", type=str, required=True, help="data type: real or syn (for synthetic data)")

    args = parser.parse_args()

    # Traverse all files in the input directory
    for filename in os.listdir(args.input_dir):

        # If it doesn't exist, create it
        os.makedirs(args.output_dir, exist_ok=True)
        filepath = os.path.join(args.input_dir, filename)
        # If it is a file, process it

        if os.path.isfile(filepath) and filename.endswith('.csv'):
            logger.info("Processing file {}".format(filepath))
            # remove duplicates
            df = pd.read_csv(filepath).drop_duplicates(subset=['Eq_Label', 'Datum', 'UOM'], keep='first')
            df = match_quants(df)

            if args.data == 'syn':
                df_posc = pd.read_csv(args.posc, encoding='cp1252')
                df_posc['label'] = df_posc['label'].str.lower()
                choices = df_posc['label'].unique().tolist()

            else:
                choices = df[args.entity].unique().tolist()

            df = add_choices(df, args.entity, choices, args.n)

            df = add_labels(df, args.entity)

            print('shuffling data...')
            df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows

            out_file = args.output_dir + '/' 'mcq' + str(args.n) + '_' + str(args.data) + '_' \
                            + args.entity + '.csv'

            df.to_csv(out_file, index=False)

            logger.info("Saving dataframes to {}".format(out_file))


if __name__ == '__main__':

    start = time.time()
    main()
    end = time.time()

    elapsed_time = (end - start) / 60
    logger.info("Total elapsed minutes: {}".format(elapsed_time))
