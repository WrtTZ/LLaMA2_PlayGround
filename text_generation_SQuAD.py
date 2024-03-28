import fire
from llama import Llama
from typing import List
import numpy as np
import csv
import re
import pandas as pd
from datasets import load_dataset


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.9,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 1,
    max_batch_size: int = 4,
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


    # clear the output csv file
    filename = "results_csv/SQuAD_generated_output.csv"
    with open(filename, 'wb') as f:
        f.truncate()

    data = load_dataset("squad")

    for index in range(0, 2000):
        print(f"\n\nindex: {index}\n\n")

        answer = data["validation"][index]["answers"]["text"][0]
        answer = answer.replace(";", ",")
        question = data["validation"][index]["context"] + " " + data["validation"][index]["question"]
        answer_words = re.findall(r'[,\.\']|[^,\.\s]+', answer)
        ans_len = len(answer_words)

        # generating tokens and record confidence
        for idx in range(0, ans_len):
            print("\n==================================\n")
            prompts = [question + ' ' + ' '.join(answer_words[:idx])]
            print("prompts: ", prompts)
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                logprobs=True
            )

            for prompt, result in zip(prompts, results):
                print(prompt)
                print(f"> {result['generation']}")

            print("\n tokens and probabilities \n")
            for result in results:
                if (len(result['tokens']) == 0):
                    continue
                if (result['tokens'][0] == "›"):
                    continue
                for t, p in zip(result['tokens'], result['logprobs']):
                    print(f"{t}--{np.exp(p)}\n")
                print("\n==================================\n")

                # store results into csv file
                with open(filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows([[result['tokens'][0], np.exp(result['logprobs'][0]), 
                                    answer_words[idx], result['tokens'][0]==answer_words[idx]]])


if __name__ == "__main__":
    fire.Fire(main)
