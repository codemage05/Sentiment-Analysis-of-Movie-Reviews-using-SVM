from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(y_test, y_pred, output_dir='images'):
    """
    Prints metrics and saves the confusion matrix plot.
    """
    # Print Metrics
    print("\n" + "-"*50)
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Save plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    print(f"\nConfusion matrix saved to {output_dir}/confusion_matrix.png")