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
size: int = 3

def Calculate_ECE(bins_num=10, df=None):
    intervals = np.linspace(0, 1, bins_num + 1)
    total_len = len(df)
    ECE = 0
    accuracy_bin = []

    # for each bin, calculate the weighted difference between accuracy and confidence
    for i in range(0, bins_num):
        con1 = (df["confidence"] <= intervals[i + 1]) & (df["confidence"] > intervals[i])
        temp = df[con1] # all entries in the current (i-th) bin
        bin_len = len(temp) # number of entries in the bin
        if bin_len == 0:
            accuracy_bin.append(0)
            continue
        correct_num = len(temp[temp["TF"]==True]) # number of correct generated tokens
        weight = bin_len / total_len
        accuracy = correct_num / bin_len
        confidence = temp["confidence"].mean()
        accuracy_bin.append(accuracy)

        calibration_error = abs(accuracy - confidence)
        weighted_ECE = weight * calibration_error
        ECE += weighted_ECE
        print(f"bin {i}: confidence-{confidence:.6} accuracy-{accuracy:.6} raw-{calibration_error:.6} weight-{weight:.6} weighted-{weighted_ECE:.6}")
    
    # plot
    fig, ax = plt.subplots()
    # bar chart
    bar_widths = np.diff(intervals)
    x_values = intervals
    bars = ax.bar(x_values[:-1], accuracy_bin, width=bar_widths, align='edge')
    # Plot the line y=x
    ax.plot(intervals, intervals, color='black', linestyle='--', label='y = x')
    # distinguish each bin
    bar_heights = [bar.get_height() for bar in bars]
    for x_position, bar_height in zip(intervals, bar_heights):
        ax.vlines(x=x_position, ymin=0, ymax=bar_height, color='black', linewidth=1)


    # set labels
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('ECE of LLaMA2')
    ax.text(0.01, 0.9, f"ECE={ECE:.4}", fontsize=8, color="black")
    plt.savefig(f'results_plot/LFQA_ECE_{str(seed)}_{str(size)}.png', dpi=500)

    print(f"ECE: {ECE:.4}")
        

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 1,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 1,
    max_batch_size: int = 5,
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

    col_names = ["generated", "groud-truth", "confidence", "TF"]
    all_results = pd.DataFrame(columns=col_names)

    data_path = "dataset/LFQA.json"
    df = pd.read_json(data_path, lines=True)
    data = df[['title', 'answers']]
    sampled_data = data.sample(n=size, random_state=seed)

    for index, row in sampled_data.iterrows():
        answer = row['answers']['text'][0]
        answer = answer.replace(";", ",")
        question = row['title']
        answer_words = re.findall(r'[,\.\']|[^,\.\s]+', answer)
        ans_len = len(answer_words)
        print(f"\nIndex: {index}, ans_len: {ans_len}\n")

        # generating tokens and record confidence
        prompts = [question]
        prompt_len = min(ans_len, ans_len)
        print(f"prompts: {prompts}")
        for idx in range(0, prompt_len):
            print(f"prompts: {prompts}")
            logits, next_token, results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=1,
                top_p=top_p,
                logprobs=True
            )

            record = {}
            for result in results:
                
                print(f"tokens and probabilities: length{len(result['tokens'])}, {len(result['logprobs'])}")
                for t, p in zip(result['tokens'], result['logprobs']):
                    print(f"{t}--{p}")
                print("==================================\n")
                
                record["generated"] = result['tokens'][0]
                record["groud-truth"] = answer_words[idx]
                record["confidence"] = result['logprobs'][0]
                record["TF"] = (record["generated"] == record["groud-truth"])
            all_results = pd.concat([all_results, pd.DataFrame([record])], ignore_index=True)

            prompts = [prompts[0] + result['generation']]

    # output the results into csv file
    filename = f"results_csv/LFQA_results_{str(seed)}_{str(size)}.csv"
    with open(filename, 'wb') as f:
        f.truncate()
    all_results.to_csv(filename, index=False)   

    # Calculate_ECE(bins_num=10, df=all_results)

if __name__ == "__main__":
    fire.Fire(main)
