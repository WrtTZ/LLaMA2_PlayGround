import fire
from llama import Llama
from typing import List
import numpy as np
import csv
import re
import pandas as pd

# ====================================================================================================
# function that takes in a generation results and calculate the ECE of each bin
def Calculate_ECE(bins_num=10, csv_file_path="results_csv/temp_results.csv"):
    # getting the results and import as dataframe
    col_names = ["generated", "confidence", "groud-truth", "accurate"]
    df = pd.read_csv(csv_file_path, header=None)
    df.columns= col_names

    intervals = np.linspace(0, 1, bins_num + 1)
    ECE_bin = []

    # for each bin, calculate the weighted difference between accuracy and confidence
    for i in range(0, bins_num):
        con1 = (df["confidence"] <= intervals[i + 1]) & (df["confidence"] > intervals[i])
        temp = df[con1] # all entries in the current (i-th) bin
        bin_len = len(temp) # number of entries in the bin
        if bin_len == 0:
            print("XXVXXV")
            ECE_bin.append(2)
            continue
        correct_num = len(temp[temp["accurate"]==True]) # number of correct generated tokens
        accuracy = correct_num / bin_len
        confidence = temp["confidence"].mean()
        calibration_error = abs(accuracy - confidence)
        ECE_bin.append(calibration_error)
    
    return ECE_bin
# ====================================================================================================

# ====================================================================================================
# fucntion that runs a single generation
def generate(
    generator = None,
    temperature: float = 1,
    top_p: float = 0.9,
    max_gen_len: int = 1,
):

    # clear the output csv file
    filename = f"results_csv/temp_results_{temperature}.csv"
    with open(filename, 'wb') as f:
        f.truncate()

    data_path = "dataset/LFQA.json"
    df = pd.read_json(data_path, lines=True)
    data = df[['title', 'answers']]

    for index, row in data.iterrows():
        print(f"\n\nindex: {index}\n\n")
        if index == 10:
            break

        answer = row['answers']['text'][0]
        answer = answer.replace(";", ",")
        question = row['title']
        answer_words = re.findall(r'[,\.\']|[^,\.\s]+', answer)
        ans_len = len(answer_words)

        # generating tokens and record confidence
        for idx in range(0, ans_len):
            prompts = [question + ' ' + ' '.join(answer_words[:idx])]
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                logprobs=True
            )

            for result in results:
                if (len(result['tokens']) == 0):
                    continue
                if (result['tokens'][0] == "›"):
                    continue
                # store results into csv file
                with open(filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows([[result['tokens'][0], result['logprobs'][0], 
                                    answer_words[idx], result['tokens'][0]==answer_words[idx]]])
# ====================================================================================================


# ====================================================================================================
# main function for temperature scaling
# should return a list of T' represents adjusted temperature for each bin
def temperature_scaling():
    all_values = {}

    # building the model for inference, once since the weights are not modified
    generator = Llama.build(
        ckpt_dir="llama/llama-2-7b",
        tokenizer_path="llama/tokenizer.model",
        max_seq_len=4096,
        max_batch_size=4,
    )

    candidates = np.linspace(0, 2, 21)
    candidates = candidates[1:]
    for i in candidates:
        t = round(i, 1)
        generate(generator, temperature=t)
        all_values[t] = Calculate_ECE(bins_num=10, csv_file_path=f"results_csv/temp_results_{t}.csv")

    df = pd.DataFrame(all_values)
    df['MaxValue'] = df.min(axis=1)
    df['MaxColumn'] = df.idxmin(axis=1)
    print("=======FINAL FINAL VALUES:=======")
    print(df)
    adjusted_temperature = df["MaxColumn"]
    output_path = "llama/adjusted_temperature.csv"
    adjusted_temperature.to_csv(output_path)
    df.to_csv("llama/adjusted_temperature_backup.csv")


# ====================================================================================================

if __name__ == "__main__":
    fire.Fire(temperature_scaling)