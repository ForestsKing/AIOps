from matplotlib import pyplot as plt


def plot_loss(train_loss, valid_loss, path):
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(valid_loss, label="Valid Loss")
    plt.title("Losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyLoss")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
