import fire
from llama import Llama
from typing import List
import numpy as np
import csv
import re
import pandas as pd
import torch

# **********************************************************************
# Output of this file:
# (1) adjusted temperatueres
#       - file path: f"llama/adjusted_temperature_{str(seed)}_{str(size)}.csv"
# (2) all the results, including tokens and their confidence under different temperatures:
#       - file path: f"results_csv/temp_results_{str(seed)}_{str(size)}.csv"
# (3) ECEs of each bin with each temperatures based on the all the results
#       - at the end of the *log.out* file
# **********************************************************************

seed: int = 42
size: int = 300

# ====================================================================================================
# function that takes in a generation results and calculate the ECE of each bin
def Calculate_ECE(bins_num=10, results=None):
    candidates = np.linspace(0.6, 1.8, 25)

    # for each bin, calculate the weighted difference between accuracy and confidence
    intervals = np.linspace(0, 1, bins_num + 1)
    ECE_bin = {}
    for i in range(0, bins_num):
        con1 = (results["confidence"] <= intervals[i + 1]) & (results["confidence"] > intervals[i])
        temp = results[con1] # all entries in the current (i-th) bin
        total_len = len(temp) # number of entries in the bin
        temp_bin = {}
        print(f"\n\n {i}:\n")
        for t in candidates:
            ECE = 0
            t = round(t, 2)
            for j in range(0, bins_num):
                con2 = (temp["confidence" + str(t)] <= intervals[j + 1]) & (temp["confidence" + str(t)] > intervals[j])
                temp2 = temp[con2]
                bin_len = len(temp2) # number of entries in the bin
                if bin_len == 0:
                    print(f"bin {j}:BIN_LENGTH_0")
                    continue
                correct_num = len(temp2[temp2["TF"]==True]) # number of correct generated tokens
                weight = bin_len / total_len
                accuracy = correct_num / bin_len
                confidence = temp2["confidence" + str(t)].mean()
                calibration_error = abs(accuracy - confidence)
                weighted_ECE = weight * calibration_error
                ECE += weighted_ECE
                print(f"bin {j}: confidence-{confidence:.6} accuracy-{accuracy:.6} raw-{calibration_error:.6} weight-{weight:.6} weighted-{weighted_ECE:.6}")
            temp_bin[t] = ECE
            print(f"{t}: {ECE}")
        
        ECE_bin[i] = [min(temp_bin, key=lambda k: temp_bin[k])]
        
    return ECE_bin
# ====================================================================================================

# ====================================================================================================
def generate(
    generator = None,
    temperature: float = 1,
    top_p: float = 0.9,
    max_gen_len: int = 1,
):

    data_path = "dataset/LFQA.json"
    df = pd.read_json(data_path, lines=True)
    data = df[['title', 'answers']]

    sampled_data = data.sample(n=size, random_state=seed)
    candidates = np.linspace(0.6, 1.8, 25)

    col_names = ["generated", "groud-truth", "confidence", "TF"]
    for x in candidates:
        t = str(round(x, 2))
        col_names.append("confidence" + t)
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
            logits, next_token, results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=1,
                top_p=top_p,
                logprobs=True
            )

            record = {}
            # original results
            for result in results:
                if (len(result['tokens']) == 0):
                    continue

                record["generated"] = result['tokens'][0]
                record["groud-truth"] = answer_words[idx + 1]
                record["confidence"] = result['logprobs'][0]
                record["TF"] = (record["generated"] == record["groud-truth"])
                
                # results with different temperatures
                for i in candidates:
                    t = round(i, 2)
                    record["confidence" + str(t)] = torch.softmax(logits/t, dim=-1)[0][next_token].tolist()[0]

            all_results = pd.concat([all_results, pd.DataFrame([record])], ignore_index=True)

    return all_results
# ====================================================================================================


# ====================================================================================================
# main function for temperature scaling
# should return a list of T' represents adjusted temperature for each bin
def temperature_scaling():
    # building the model for inference, once since the weights are not modified
    generator = Llama.build(
        ckpt_dir="llama/llama-2-7b",
        tokenizer_path="llama/tokenizer.model",
        max_seq_len=2048,
        max_batch_size=4,
    )

    all_results = generate(generator)
    
    # store all_results into the csv file
    filename = f"results_csv/temp_results_{str(seed)}_{str(size)}.csv"
    with open(filename, 'wb') as f:
        f.truncate()
    all_results.to_csv(filename, index=False)

    adjusted_temperature = Calculate_ECE(bins_num=10, results=all_results)
    print(adjusted_temperature)
    df = pd.DataFrame(adjusted_temperature)
    output_path = f"llama/adjusted_temperature_{str(seed)}_{str(size)}.csv"
    df.to_csv(output_path)

# ====================================================================================================

if __name__ == "__main__":
    fire.Fire(temperature_scaling)