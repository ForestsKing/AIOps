from matplotlib import pyplot as plt


def plot_loss(forecast, reconstruct, total, path):
    plt.figure()
    plt.plot(forecast, label="Forecast loss")
    plt.plot(reconstruct, label="Reconstruct loss")
    plt.plot(total, label="Total loss")
    plt.title("Losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
