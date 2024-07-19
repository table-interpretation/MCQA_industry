from ast import literal_eval
from collections import defaultdict
import pandas as pd

def multi_choice(choices):
    # Listing the choices with letters
    choice_symbol = {}

    for count, value in enumerate(choices):
        letter = chr(ord('A') + count)
        choice_symbol[value] = letter

    return choice_symbol

def prompt_multi(prompt, choices):
    # Putting the prompt together: Question + choices + Answer:
    choices = multi_choice(choices)

    prompt += '\n'
    for choice, symbol in choices.items():
        prompt += symbol + '. ' + choice + '\n'
    prompt += "Answer: "
    return prompt


def generate_examples(df_train, k, entity, fchoice):
    # Generating few-shot examples
    ex_quants = list(dict(df_train[entity].value_counts(ascending=False)))[:k]
    shots = defaultdict(list)
    for i in range(len(df_train)):

        quant = df_train[entity].values[i]
        question = df_train['Question'].values[i]
        choices = multi_choice(literal_eval(df_train[fchoice].values[i]))

        if quant in ex_quants:  # quant in test may not be in train

            prompt = prompt_multi(question, choices)
            prompt += choices[quant]
            shots[quant].append(prompt)
            if len(shots) == k:
                break
    examples = [prlist[0] for _, prlist in shots.items()]

    return examples


def get_posc_def(file='../data/results_props.csv'):
    # Loading the extracted quantities and definitions from POSC Caesar
    quant_def_dict = {}
    df = pd.read_csv(file, encoding='cp1252')
    for i in range(len(df)):
        quant = df.loc[:, 'label'][i].lower()
        definition = df.loc[:, 'definition'][i].lower().replace('.', '')

        quant_def_dict[quant.strip()] = definition

    return quant_def_dict


def add_quant_def(df, posc_props):
    # Adding POSC Ceasar definitions to the quantity choices.
    quant_def = get_posc_def(posc_props)
    for i in range(len(df)):
        quant = df.loc[:, 'Quant'][i]
        if 'max' in quant:
            max_quant = ' '.join(x for x in quant.split() if x != 'max')
            max_def = quant_def.get(max_quant, '')
            quant_def[quant] = max_def + ' at maximum'
        if 'min' in quant:
            min_quant = ' '.join(x for x in quant.split() if x != 'min')
            min_def = quant_def.get(min_quant, '')
            quant_def[quant] = min_def + ' at minimum'

    df['final_choices_w_def'] = ''
    df['Quant_w_Def'] = ''
    for i in range(len(df)):
        quant = df.loc[:, 'Quant'][i]
        df['Quant_w_Def'].values[i] = quant + ': ' + quant_def.get(quant, '')
        final_choices = literal_eval(df.loc[:, 'final_choices'][i])

        fch_def = []
        for fch in final_choices:
            fch += ': ' + quant_def.get(fch, '')
            fch_def.append(fch)

        df['final_choices_w_def'].values[i] = fch_def

    return df

