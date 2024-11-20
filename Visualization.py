from pathlib import Path

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_confusion_matrix(y_true, y_pred, save_path = Path(__file__).parent):
    '''
    :param y_true: A list or array of the label of test dataset
    :param y_pred: A list or array of the prediction of test dateset
    :param save_path: The save path for confusion matrix and classification report
    '''
    # classification report
    report = classification_report(y_true, y_pred, target_names=['negative', 'neutral', 'positive'], output_dict=True)
    print("Classification Report:\n", report)
    df = pd.DataFrame(report).transpose()
    df.to_csv(save_path / "classification_report.csv", index=True)
    # confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['negative', 'neutral', 'positive'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(save_path / "confusion_matrix.png")
    plt.show()

def plot_loss(train_loss=None, valid_loss = None, save_path = Path(__file__).parent):
    if train_loss is None:
        train_loss = []
    if valid_loss is None:
        valid_loss = []

    # Seaborn style
    sns.set(style="darkgrid", context="talk", palette="Set2")
    plt.figure(figsize=(10, 7))

    data = {
        "Epochs": list(range(len(train_loss))) + list(range(len(valid_loss))),
        "Loss": train_loss + valid_loss,
        "Type": ["Training"] * len(train_loss) + ["Validation"] * len(valid_loss)
    }
    df = pd.DataFrame(data)

    # plot curve
    sns.lineplot(data=df, x="Epochs", y="Loss", hue="Type", style="Type", markers=True, dashes=False, linewidth=2.5)

    # min_value
    if len(train_loss) > 0:
        plt.axhline(y=min(train_loss), color=sns.color_palette("Set2")[0], linestyle='--', linewidth=2,
                    label='Min Training Loss')
    if len(valid_loss) > 0:
        plt.axhline(y=min(valid_loss), color=sns.color_palette("Set2")[1], linestyle='--', linewidth=2,
                    label='Min Validation Loss')

    plt.title('Training vs Validation Loss Over Epochs', fontsize=20, fontweight='bold', color="black")
    plt.xlabel('Epochs', fontsize=16, fontweight='bold', color="darkgreen")
    plt.ylabel('Loss', fontsize=16, fontweight='bold', color="darkgreen")

    plt.xticks(ticks=[i for i in list(range(len(train_loss))) if i % 5 == 0], fontsize=14, color="black")
    plt.yticks(fontsize=14, color="black")

    plt.grid(color="lightgray", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(title="Loss Type", fontsize=12, title_fontsize=14, loc='upper right', shadow=True)

    plt.tight_layout()
    plt.savefig(save_path / "loss.png")
    plt.show()


if __name__ == '__main__':
    train_loss = [0.9 - 0.04 * i for i in range(20)]
    valid_loss = [0.95 - 0.03 * i + (0.01 * (-1) ** i) for i in range(20)]
    plot_loss(train_loss=train_loss, valid_loss=valid_loss)