import fire
from llama import Llama
from typing import List
import numpy as np
import csv
import re
import pandas as pd
import torch
import matplotlib.pyplot as plt

seed: int = 41
size: int = 30
mode: int = 1

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def main(
    ckpt_dir: str = "llama/llama-2-7b",
    tokenizer_path: str = "llama/tokenizer.model",
    temperature: float = 1,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_gen_len: int = 32,
    max_batch_size: int = 5,
    mode: int = mode,
    beam_length: int = 0,
    temperature_file_path: str = f"llama/adjusted_temperature_42_300.csv"
):

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

    ground_truth_probability_list = []
    greedy_probability_list = []
    greedy_probability_list_2 = []
    bs_generation_logprobs_list = []
    ns_probability_list = []
    ns_probability_list_2 = []
    
    
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
        
        ground_truth_probability = float(1.0)
        prompt = [encoded_question]
        
        # calculate the ground-truth probability
        for idx in range(0, ans_len):
            next_idx = encoded_answer[idx]
            
            raw_logits, next_token, generation_tokens, generation_logprobs = generator.generate(
                prompt_tokens=prompt,
                max_gen_len=1,
                temperature=temperature,
                top_p=top_p,
                logprobs=True,
                echo=False,
                mode=1,
                beam_length=beam_length,
                temperature_file_path=temperature_file_path
            )
            probs = torch.softmax(raw_logits, dim=-1)
            probs_next = probs[0][next_idx]
            # ground_truth_probability *= probs_next
            ground_truth_probability_list.append(probs_next.item())
            greedy_probability_list.append(generation_logprobs[0])
            id_ns = sample_top_p(probs, 0.9)
            id_ns = id_ns.reshape(-1).item()
            prob_ns = probs[0][id_ns]
            ns_probability_list.append(prob_ns.item())
            
            prompt[0].append(encoded_answer[idx])
            
        # calculate the greedy probability and the nucleus sampling probability
        temp_greedy_token_list = []
        temp_greedy_prob_list = []
        temp_ns_token_list = []
        temp_ns_prob_list = []
        prompt = [generator.tokenizer.encode(question, bos=True, eos=False)]
        for idx in range(0, ans_len):
            raw_logits, next_token, generation_tokens, generation_logprobs = generator.generate(
                prompt_tokens=prompt,
                max_gen_len=1,
                temperature=temperature,
                top_p=top_p,
                logprobs=True,
                echo=False,
                mode=1,
                beam_length=beam_length,
                temperature_file_path=temperature_file_path
            )
            greedy_probability_list_2.append(generation_logprobs[0][0])
            temp_greedy_token_list.append(generation_tokens[0][0])
            temp_greedy_prob_list.append(generation_logprobs[0][0])
            
            probs = torch.softmax(raw_logits, dim=-1)
            id_ns = sample_top_p(probs, 0.9)
            id_ns = id_ns.reshape(-1).item()
            prob_ns = probs[0][id_ns]
            ns_probability_list_2.append(prob_ns.item())
            temp_ns_token_list.append(id_ns)
            temp_ns_prob_list.append(prob_ns.item())
            
            prompt[0].append(generation_tokens[0][0])
            
        print("===============Greedy===================\n")
        print(generator.tokenizer.decode(temp_greedy_token_list))
        print(temp_greedy_prob_list)
        print("===============Nucleus Sampling===================\n")
        print(generator.tokenizer.decode(temp_ns_token_list))
        print(temp_ns_prob_list)

        # calculate the beam search probability
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
        
        print("===============BS===================\n")
        print(generator.tokenizer.decode(generation_tokens))
        print(bs_generation_logprobs)
        
        bs_generation_logprobs_list = bs_generation_logprobs_list + bs_generation_logprobs
        print("================================\n")
    
    col_names = ["ground_truth", "ground_truth_greedy", "ground_truth_nucleus_sampling", "greedy", "nucleus_sampling", "beam_search"]
    all_results = pd.DataFrame(columns=col_names)
    
    print(ns_probability_list)
    print(ns_probability_list_2)
    for j in range(0, len(bs_generation_logprobs_list)):
        record = {}
        record["ground_truth"] = ground_truth_probability_list[j]
        record["ground_truth_greedy"] = greedy_probability_list[j][0]
        record["ground_truth_nucleus_sampling"] = ns_probability_list[j]
        record["greedy"] = greedy_probability_list_2[j]
        record["nucleus_sampling"] = ns_probability_list_2[j]
        record["beam_search"] = bs_generation_logprobs_list[j]
        
        all_results = pd.concat([all_results, pd.DataFrame([record])], ignore_index=True)
        
    # output the results into csv file
    filename = f"results_csv/ground_truth_probability_{str(seed)}_{str(size)}.csv"
    with open(filename, 'wb') as f:
        f.truncate()
    all_results.to_csv(filename, index=False)
    
if __name__ == "__main__":
    fire.Fire(main)
