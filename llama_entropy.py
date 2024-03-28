import fire
from llama import Llama
from typing import List
import numpy as np
import csv
import re
import pandas as pd
import torch
import tensor_parallel as tp
import matplotlib.pyplot as plt

# **********************************************************************
# Output of this file:
# (1) raw results:
#       - file path: f"results_csv/LFQA_results_{str(seed)}_{str(size)}.csv"
# (2) ECE plot:
#       - file path: f'results_plot/LFQA_ECE_{str(seed)}_{str(size)}.png
# (3) ECE of each bin:
#       - at the end of the *log.out* file
# **********************************************************************

seed: int = 41
size: int = 100
mode: int = 1

def entropy(probabilities):
    probabilities = np.array(probabilities)
    entropy_value = -np.sum(probabilities * np.log2(probabilities, where=(probabilities > 0)))
    return entropy_value

def main(
    ckpt_dir: str = "llama/llama-2-7b",
    tokenizer_path: str = "llama/tokenizer.model",
    temperature: float = 1,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 32,
    max_batch_size: int = 5,
    mode: int = mode,
    beam_length: int = 0,
    temperature_file_path: str = f"llama/adjusted_temperature_42_300.csv"
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    data_path = "dataset/LFQA.json"
    df = pd.read_json(data_path, lines=True)
    data = df[['title', 'answers']]
    sampled_data = data.sample(n=size, random_state=seed)

    entropy_list = []
    all_len = 0
    for index, row in sampled_data.iterrows():
        answer = row['answers']['text'][0]
        answer = answer.replace(";", ",")
        question = row['title']
        answer_words = re.findall(r'[,\.\']|[^,\.\s]+', answer)
        ans_len = len(answer_words)
        all_len += ans_len
        print(f"\nIndex: {index}, ans_len: {ans_len}\n")

        # generating tokens and record confidence
        prompts = [question]
        prompt_len = min(ans_len, ans_len)
        for idx in range(0, prompt_len):
            print(f"prompts: {prompts}")
            logits, next_token, results = generator.text_completion(
                prompts,
                max_gen_len=1,
                temperature=1,
                top_p=top_p,
                logprobs=True,
                mode=1
            )

            # Get the indices of the top 42 values
            # print(f"logits: {logits}")
            top_indices = torch.topk(logits[0], k=42).indices
            # print(f"top_indices: {top_indices}")
            # print(type(top_indices))

            # Extract the top 42 values using the indices
            top_values = logits[0][top_indices]
            # print(f"top_values : {top_values}")
            # print(type(top_values))

            # Normalize the values by dividing each element by the sum of all 42 elements
            normalized_values = torch.softmax(top_values, dim=-1)
            # print(f"normalized_values: {normalized_values}")
            # print(type(normalized_values))

            # Convert the PyTorch tensor to a Python list
            normalized_list = normalized_values.tolist()
            # print(f"normalized_list: {normalized_list}")
            # print(type(normalized_list))
            
            temp_entropy = entropy(normalized_list)
            # print(f"temp_entropy: {temp_entropy}")
            # print(type(temp_entropy))
            entropy_list.append(temp_entropy)
            # print(entropy_list)
            # print(type(entropy_list[0]))
            
            for result in results:
                print(f"tokens and probabilities: length{len(result['tokens'])}, {len(result['logprobs'])}")
                for t, p in zip(result['tokens'], result['logprobs']):
                    print(f"{t}--{p}")
                print("==================================\n")


            prompts = [prompts[0] + ' ' + answer_words[idx]]

    # print(entropy_list)
    print(f"mean: {np.mean(entropy_list)}, var: {np.var(entropy_list)}, all_len: {all_len}")

if __name__ == "__main__":
    fire.Fire(main)
