import time
import os.path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from ast import literal_eval
import argparse

import openai
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tenacity import retry, wait_fixed, stop_after_attempt
from utils import generate_examples, prompt_multi
#from gpt_utils import read_gpt_config

def init_llama_model():
    """
    Initialization for the llama2 pipeline
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", device_map='auto')
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map='auto')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pipeline = transformers.pipeline("text-generation",
                                     model=model,
                                     tokenizer=tokenizer,
                                     torch_dtype=torch.float16,
                                     device_map='auto',
                                     )
    return pipeline

def get_llama_responce(pipeline, prompt, out_dir, out_file):
    print("Start Llama")
    responce = pipeline(prompt,
                        temperature=0.2,
                        do_sample=True,
                        top_k=10,
                        num_return_sequences=5,
                        max_length=4096)

    textfile = out_file.replace('csv', 'txt')
    with open(os.path.join(out_dir, textfile), 'w') as file:
        file.write("{}".format(responce))
        file.write("\n")

    return responce


@retry(wait=wait_fixed(60), stop=stop_after_attempt(5))
def run_gpt(prompt, llm, out_dir, kshots, r, cot=False):
    """
    Call to OpenAI API
    :param prompt:
    :param llm:
    :param out_dir:
    :return:
    """
    # Add own function for reading the configurations for accessing the Open AI API.
    # read_gpt_config(llm)
    try:
        if llm == "gpt-35-turbo-instruct":
            response = openai.Completion.create(
                engine=llm,
                prompt=prompt,
                temperature=0,
                max_tokens=5,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0)

        elif llm in ["gpt-35-turbo", "gpt-35-turbo-16k", "gpt-4", "gpt-4o-2024-05-13"]:

            content_sys = "The following are multiple choice questions about the physical properties " \
                          "of industrial plant equipment.""The right answer is included."
            if cot:
                content_sys = content_sys + "You should reason in a step-by-step manner as to get the right answer."
                prompt = prompt + "Let's think step by step: "

                response = openai.ChatCompletion.create(
                    engine=llm,
                    messages=[{"role": "system", "content": content_sys},
                              {"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1000,
                    top_p=0.5,
                    frequency_penalty=0,
                    presence_penalty=0,
                    seed=11)

            else:
                content_sys = content_sys + "You should directly answer the question by choosing the correct option."
                response = openai.ChatCompletion.create(
                    engine=llm,
                    messages=[{"role": "system", "content": content_sys}, {"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1000,
                    top_p=0.5,
                    frequency_penalty=0,
                    presence_penalty=0,
                    seed=11)

    except openai.error.Timeout as e:
        # Handle timeout error, e.g. retry or log
        print(f"OpenAI API request timed out: {e}")
        time.sleep(2)
        pass

    except openai.error.APIError as e:
        # Handle API error, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        time.sleep(2)
        pass

    except openai.error.APIConnectionError as e:
        # Handle connection error, e.g. check network or log
        print(f"OpenAI API request failed to connect: {e}")
        time.sleep(2)
        pass

    except openai.error.InvalidRequestError as e:
        # Handle invalid request error, e.g. validate parameters or log
        print(f"OpenAI API request, {prompt} was invalid: {e}")
        time.sleep(2)
        pass

    except openai.error.AuthenticationError as e:
        # Handle authentication error, e.g. check credentials or log
        print(f"OpenAI API request was not authorized: {e}")
        time.sleep(2)
        pass

    except openai.error.PermissionError as e:
        # Handle permission error, e.g. check scope or log
        print(f"OpenAI API request was not permitted: {e}")
        time.sleep(2)
        pass

    except openai.error.RateLimitError as e:
        # Handle rate limit error, e.g. wait or log
        print(f"OpenAI API request exceeded rate limit: {e}")
        time.sleep(20)
        pass

    else:
        print(llm + " API call succeeded.")

    if len(response) > 0:
        choices = response["choices"][0]
        try:
            if llm == "gpt-35-turbo-instruct":
                ans = choices["text"]
            else:
                ans = choices["message"]["content"]

            with open(os.path.join(out_dir, 'log_file_{}_{}_{}_syn.txt'.format(llm, kshots, r)), 'a') as file:
                file.write("{}".format(ans))
                file.write("\n")

            with open(os.path.join(out_dir, 'log_prompt_{}_cot_{}_syn.txt'.format(llm, cot)), 'a') as file:
                file.write("{}".format(prompt))
                file.write("\n")

            if 'sorry' in ans.lower() or 'none' in ans.lower():
                ans = 'unknown'

        except:
            with open(os.path.join(out_dir, 'log_file_{}_{}_{}.txt'.format(llm, kshots, r)), 'a') as file:
                file.write("{}".format(response["choices"]))
                file.write("\n")

    return ans


def run(test_file, train_file, kshot, llm, out_dir, out_file, r, w_def="y", cot=False):

    print('reading test file {}'.format(test_file))
    df_test = pd.read_csv(test_file)
    print('reading train file  {}'.format(train_file))
    df_train = pd.read_csv(train_file)

    df_test['Prompt'] = ''
    df_test['Final_Answer'] = ''
    df_test['Score'] = ''

    pbar_first = tqdm(range(len(df_test)), desc="test progress", position=0)

    for i in pbar_first:
        # not adding defs as part of the prompt. Only adding to the few-shot examples
        if w_def == "y":
            entity = 'Quant_w_Def'
            fchoice = 'final_choices_w_def'
        else:
            entity = 'Quant'
            fchoice = 'final_choices'

        if kshot > 0:
            shots = generate_examples(df_train, kshot, entity, fchoice)
        else:
            shots = [""]

        prompt = df_test['Question'].values[i]
        for shot in shots:
            prompt = shot + '\n' + prompt
        prompt = prompt_multi(prompt, literal_eval(df_test['final_choices'].values[i]))

        if llm == "llama-2-7b":
            pipeline = init_llama_model()
            ans = get_llama_responce(pipeline, prompt, out_dir, out_file)
        else:
            ans = run_gpt(prompt, llm, out_dir, r, cot)

        ######################### Parsing of the final answer ##########################################################
        if cot:
            position_keyword = ans.find("Given")
            position_there = ans.find("Therefore")
            position_based = ans.find("Based")

            if position_keyword != -1:
                new_answer = "Given " + ans[position_keyword+len("Given"):].strip()
            elif position_there != -1:
                new_answer = "Therefore " + ans[position_there + len("Therefore"):].strip()
            elif position_based != -1:
                new_answer = "Based " + ans[position_based + len("Based"):].strip()
            else:
                new_answer = ans

            ans = new_answer
        df_test['Prompt'].values[i] = prompt
        df_test['Final_Answer'].values[i] = ans
        gold = chr(df_test['label'].values[i] + 65)
        df_test['Score'].values[i] = 1 if gold in df_test['Final_Answer'].values[i] else 0

    return df_test


def main(test_file, train_file, kshot, output_dir, llm, r, w_def="y", cot=True):
    # record start time
    start = time.time()

    if cot and w_def == "y":
        if "syn" in test_file:
            train_file = "../data/syn/syn_train_defs.csv"
        else:
            train_file = "../data/realworld/realworld_train_defs.csv"

    elif cot and w_def == "n":
        if "syn" in test_file:
            train_file = "../data/syn/syn_train.csv"
        else:
            train_file = "../data/realworld/realworld_train.csv"

    out_dir = output_dir + '/' + llm + '/' + str(kshot) + 'shot'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if w_def == 'y':
        out_file = out_dir + '/' + 'result_k' + str(kshot) + '_wdef_' + test_file.split('/')[-1]
    else:
        out_file = out_dir + '/' + 'result_k' + str(kshot) + '_' + test_file.split('/')[-1]

    df_test = run(test_file, train_file, kshot, llm, out_dir, out_file, r, w_def, cot)

    df_test.to_csv(out_file, index=False)
    end = time.time()
    accuracy = df_test['Score'].mean()

    elapsed_time = (end - start) / 60
    textfile = out_file.replace(".csv", ".txt")

    # write output to a text file textfile
    with open(textfile, 'w') as f:
        f.write('Model: ' + llm + '\n')
        f.write('Train file: ' + train_file + '\n')
        f.write('Test file: ' + test_file + '\n')
        f.write('K-shot: ' + str(kshot) + '\n')
        f.write('Number of questions: ' + str(len(df_test)) + '\n')
        f.write('accuracy: ' + str(accuracy) + '\n')
        f.write('Elapsed time: ' + str(elapsed_time) + ' minutes\n')
    return accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #
    parser.add_argument("--train_file", default="../data/syn/syn_train.csv", type=str, required=False)
    parser.add_argument("--test_file", default="../data/syn/syn_test.csv", type=str, required=True)
    parser.add_argument("--cot", type=bool, required=True, help="ask for direct response or use CoT")
    parser.add_argument("--llm", type=str, required=True)
    #
    args = parser.parse_args()

    if args.cot:
        output_dir = "../output/cot"
        w_def = "y"
    else:
        output_dir = "../output/direct"
        w_def = "n"

    kshots = 5
    best_accuracy = 0
    best_k = 0
    for k in range(0, kshots + 1):
        accuracy = main(args.test_file, args.train_file, k, w_def, output_dir, args.llm, cot=args.cot)
        print("Accuracy:", accuracy)
        print("kshot:", k)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    print('best accuracy:', best_accuracy)
    print("best kshot:", best_k)