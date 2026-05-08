import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics():

    df = pd.read_csv("outputs/log.csv")

    plt.plot(df["epoch"], df["loss"], label="Loss")
    plt.plot(df["epoch"], df["acc"], label="Accuracy")
    plt.plot(df["epoch"], df["qwk"], label="QWK")

    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Training Curves")

    plt.savefig("outputs/training_plot.png")
    plt.show()