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
import evaluate


seed: int = 41
size: int = 1
mode: int = 1  # 1 for pure greedy, 2 for nucleus sampling, 3 for beam search, 4 for calibrated beam search
k: int = 5 # beam length

def main(
    ckpt_dir: str = "llama/llama-2-7b-chat",
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

    col_names = ["prompts", "groud-truth", "generations"]
    all_results = pd.DataFrame(columns=col_names)

    data_path = "dataset/LFQA.json"
    df = pd.read_json(data_path, lines=True)
    data = df[['title', 'answers']]
    sampled_data = data.sample(n=size, random_state=seed)

    predictions = []
    references = []

    for index, row in sampled_data.iterrows():
        answer = row['answers']['text'][0]
        question = row['title']
        encoded_answer = generator.tokenizer.encode(answer, bos=False, eos=False)
        ans_len = len(encoded_answer)
        encoded_question = generator.tokenizer.encode(question, bos=True, eos=False)
        print(f"answer: {answer}")
        # print(f"encoded_answer: {encoded_answer}")
        print(f"question: {question}")
        # print(f"encoded_question: {encoded_question}")
        print(f"Index: {index}, ans_len: {ans_len}\n")

        prompt = [generator.tokenizer.encode(question, bos=True, eos=False)]
        raw_logits, next_token, generation_tokens, bs_generation_logprobs = generator.generate(
            prompt_tokens=prompt,
            max_gen_len=ans_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=True,
            echo=False,
            mode=3,
            beam_length=5,
            temperature_file_path=temperature_file_path
        )
        
        record = {}
        record["prompts"] = [question]
        record["groud-truth"] = [answer]
        # record["sequences"] = result["tokens"]
        record["generations"] = generator.tokenizer.decode(generation_tokens)[0]
        predictions.append(generator.tokenizer.decode(generation_tokens)[0])
        references.append(answer)
        all_results = pd.concat([all_results, pd.DataFrame([record])], ignore_index=True)

    naming = ""
    if mode == 1:
        naming = "greedy"
    elif mode == 2:
        naming = "nucleus_sampling"
    elif mode == 3:
        naming = f"beam_search"
    elif mode == 4:
        naming = f"posthoc_beam_search"
    
    # output the results into csv file
    filename = f"results_csv/chat_LFQA_{naming}_results_{str(seed)}_{str(size)}.csv"
    with open(filename, 'wb') as f:
        f.truncate()
    all_results.to_csv(filename, index=False)

    # evaluate the rouge metric
    rouge = evaluate.load('rouge')
    print(f"predictions: {predictions}")
    print(f"references: {references}")
    results = rouge.compute(predictions=predictions, references=references)
    print(results)
    results["seed"] = seed
    results["size"] = size
    if mode == 3 or mode == 4:
        results["k"] = k
    results = {key: [round(value, 6)] for key, value in results.items()}
    df = pd.DataFrame(results)
    ROUGE_filename = f"results_csv/ROUGE_{naming}_LFQA.csv"
    existing_df = pd.read_csv(ROUGE_filename)
    df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(ROUGE_filename, index=False)

if __name__ == "__main__":
    fire.Fire(main)
