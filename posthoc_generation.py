import fire
from llama import Llama
from typing import List
import numpy as np
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch

# **********************************************************************
# Output of this file:
# (1) posthoc results:
#       - file path: f"results_csv/posthoc_LFQA_results_{str(seed)}_{str(size)}.csv"
# (2) ECE plot:
#       - file path: f'results_plot/posthoc_LFQA_ECE_{str(seed)}_{str(size)}.png
# (3) ECE of each bin:
#       - at the end of the *log.out* file
# **********************************************************************

# 0.04224
seed: int = 42
size: int = 300

def Calculate_ECE(bins_num=10, df=None):
    # getting the results and import as dataframe
    # col_names = ["generated", "confidence", "groud-truth", "TF"]

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
    plt.savefig(f'results_plot/posthoc_LFQA_ECE_{str(seed)}_{str(size)}.png', dpi=500)

    print(f"ECE: {ECE:.4}")

def posthoc_generation(
    ckpt_dir: str = "llama/llama-2-7b",
    tokenizer_path: str = "llama/tokenizer.model",
    temperature: float = 1,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 1,
    max_batch_size: int = 4,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # get the adjusted temperature parameters for each bin

    # ************** MANUALLY CHANGE *********************
    temperature_filename = f"llama/adjusted_temperature_42_300.csv"
    # ************** MANUALLY CHANGE *********************

    adjusted_t = pd.read_csv(temperature_filename)
    adjusted_t.drop(["Unnamed: 0"], inplace=True, axis=1)
    adjusted_t = adjusted_t.reset_index(drop=True)
    adjusted_t = pd.melt(adjusted_t, var_name='bin', value_name='t')
    adjusted_t['bin'] = adjusted_t['bin'].astype(int)
    adjusted_t['t'] = adjusted_t['t'].astype(float)

    data_path = "dataset/LFQA.json"
    df = pd.read_json(data_path, lines=True)
    data = df[['title', 'answers']]
    sampled_data = data.sample(n=size, random_state=seed)

    col_names = ["generated", "groud-truth", "confidence", "TF"]
    all_results = pd.DataFrame(columns=col_names)

    for index, row in sampled_data.iterrows():
        answer = row['answers']['text'][0]
        answer = answer.replace(";", ",")
        question = row['title']
        answer_words = re.findall(r'[,\.\']|[^,\.\s]+', answer)
        ans_len = len(answer_words)
        print(f"index: {index}, ans_len: {ans_len}")

        # generating tokens and record confidence
        prompts = [question]
        prompt_len = min(ans_len-1, 100)
        for idx in range(0, prompt_len):
            prompts = [prompts[0] + ' ' + answer_words[idx]]
            logits, next_token, original_results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=1,
                top_p=top_p,
                logprobs=True
            )
            if (len(original_results[0]['tokens']) == 0):
                    continue

            # finding the bin and adjusted temperature of the original output
            original_confidence = original_results[0]['logprobs'][0]
            intervals = np.linspace(0, 1, 11)
            adjusted_t["low"] = intervals[adjusted_t["bin"]]
            adjusted_t["high"] = intervals[adjusted_t["bin"] + 1]
            con = (original_confidence <= adjusted_t["high"]) & (original_confidence > adjusted_t["low"])
            temp_t = adjusted_t[con]
            adjusted_temperature = temp_t["t"].tolist()[0]
            
            record = {}
            record["generated"] = original_results[0]['tokens'][0]
            record["groud-truth"] = answer_words[idx + 1]
            record["TF"] = (record["generated"] == record["groud-truth"])

            if (adjusted_temperature == 1.0):
                record["confidence"] = original_results[0]['logprobs'][0]
            else:
                record["confidence"] = torch.softmax(logits/adjusted_temperature, dim=-1)[0][next_token].tolist()[0]
            
            all_results = pd.concat([all_results, pd.DataFrame([record])], ignore_index=True)

    # output the results into csv file
    filename = f"results_csv/posthoc_LFQA_results_{str(seed)}_{str(size)}.csv"
    with open(filename, 'wb') as f:
        f.truncate()
    all_results.to_csv(filename, index=False)   

    Calculate_ECE(bins_num=10, df=all_results)
    
if __name__ == "__main__":
    fire.Fire(posthoc_generation)
