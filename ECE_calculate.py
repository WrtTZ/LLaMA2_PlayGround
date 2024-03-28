import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# function to calculate the Expected Calbration Error (ECE) and generate a plot
def Calculate_ECE(bins_num=10, csv_file_path=None):
    # getting the results and import as dataframe
    df = pd.read_csv(csv_file_path, header=False)

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
        print(f"bin {i}: confidence-{confidence} accuracy-{accuracy} raw-{calibration_error} weight-{weight} weighted-{weighted_ECE}")
    
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
    plt.savefig('results_plot/posthoc_LFQA_ECE_1.png', dpi=500)

    print(f"ECE: {ECE:.4}")


if __name__ == "__main__":
    Calculate_ECE(bins_num=10, csv_file_path="results_csv/posthoc_LFQA_results_1.csv")

