# import evaluate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# rouge = evaluate.load('rouge')
# predictions = ["hello there", "general kenobi"]
# references = ["hello, there", "general kenobi long extra sequence for test"]
# results = rouge.compute(predictions=predictions, references=references)
# print(results)
# seed = 41
# size = 3
# k = 5

# df_bs = pd.read_csv(f"results_csv/LFQA_beam_5_search_results_{seed}_{size}.csv")
# df_pbs = pd.read_csv(f"results_csv/LFQA_posthoc_beam_5_search_results_{seed}_{size}.csv")
# df_gr = pd.read_csv(f"results_csv/LFQA_greedy_results_{seed}_{size}.csv")
# df_ns = pd.read_csv(f"results_csv/LFQA_nucleus_sampling_results_{seed}_{size}.csv")

# for (index_bs, row_bs), (index_pbs, row_pbs), (index_gr, row_gr), (index_ns, row_ns) in zip(df_bs.iterrows(), df_pbs.iterrows(), df_gr.iterrows(),df_ns.iterrows()):
#     print(f"prompts: {row_bs['prompts']}\n")
#     print(f"groud-truth: {row_bs['groud-truth']}\n")
#     print(f"bs_generated: {row_bs['generations']}\n")
#     print(f"pbs_generated: {row_pbs['generations']}\n")
#     print(f"gr_generated: {row_gr['generations']}\n")
#     print(f"ns_generated: {row_ns['generations']}\n")
#     print("==========================\n\n")

# rouge_bs = pd.read_csv(f"results_csv/ROUGE_beam_search_LFQA.csv")
# rouge_pbs = pd.read_csv(f"results_csv/ROUGE_posthoc_beam_search_LFQA.csv")
# rouge_gr = pd.read_csv(f"results_csv/ROUGE_greedy_LFQA.csv")
# rouge_ns = pd.read_csv(f"results_csv/ROUGE_nucleus_sampling_LFQA.csv")

# print(f"\nrouge_bs:\n{rouge_bs}\n")
# print(f"mean:\n{rouge_bs.mean()}\nstd:\n{rouge_bs.std()}")
# print(f"\nrouge_pbs:\n{rouge_pbs}\n")
# print(f"mean:\n{rouge_pbs.mean()}\nstd:\n{rouge_pbs.std()}")
# print(f"\nrouge_gr:\n{rouge_gr}\n")
# print(f"mean:\n{rouge_gr.mean()}\nstd:\n{rouge_gr.std()}")
# print(f"\nrouge_ns:\n{rouge_ns}\n")
# print(f"mean:\n{rouge_ns.mean()}\nstd:\n{rouge_ns.std()}")
# for (index_bs, row_bs), (index_pbs, row_pbs), (index_gr, row_gr), (index_ns, row_ns) in zip(rouge_bs.iterrows(), rouge_pbs.iterrows(), rouge_gr.iterrows(),rouge_ns.iterrows()):


seed = 41
size = 3
k = 5

df_bs = pd.read_csv(f"results_csv/LFQA_beam_5_search_results_{seed}_{size}.csv")
df_gr = pd.read_csv(f"results_csv/LFQA_greedy_results_{seed}_{size}.csv")
for (index_bs, row_bs), (index_gr, row_gr) in zip(df_bs.iterrows(), df_gr.iterrows()):
    print(f"prompts: {row_bs['prompts']}\n")
    print(f"groud-truth: {row_bs['groud-truth']}\n")
    print(f"bs_generated: {row_bs['generations']}\n")
    print(f"gr_generated: {row_gr['generations']}\n")
    print("==========================\n\n")

rouge_bs = pd.read_csv(f"results_csv/ROUGE_beam_search_LFQA.csv")
rouge_gr = pd.read_csv(f"results_csv/ROUGE_greedy_LFQA.csv")
print(f"\nrouge_bs:\n{rouge_bs}\n")
print(f"mean:\n{rouge_bs.mean()}\nstd:\n{rouge_bs.std()}")
print(f"\nrouge_gr:\n{rouge_gr}\n")
print(f"mean:\n{rouge_gr.mean()}\nstd:\n{rouge_gr.std()}")






