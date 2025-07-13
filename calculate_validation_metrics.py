from pathlib import Path
import paths
import functions_th as th
import  Panorama_Switching as ps
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
sys.path.append('./functions')
import ml_functions as mlfn

project_root = Path(__file__).resolve().parent
classified_healthy_path = str(project_root / "validation_data" / "classified_healthy")
classified_faulty_path = str(project_root / "validation_data" / "classified_faulty")
actual_healthy_path = str(project_root / "validation_data" / "actual_healthy")
actual_faulty_path = str(project_root / "validation_data" / "actual_faulty")

def compute_confusion_matrix_from_folders(classified_healthy_path, classified_faulty_path, actual_healthy_path, actual_faulty_path):
    """
    Returns: TP, FP, TN, FN, y_true, y_pred
    """
    classified_healthy = set(Path(classified_healthy_path).iterdir())
    classified_faulty = set(Path(classified_faulty_path).iterdir())
    actual_healthy = set(Path(actual_healthy_path).iterdir())
    actual_faulty = set(Path(actual_faulty_path).iterdir())

    classified_healthy_names = set(p.name for p in classified_healthy)
    classified_faulty_names = set(p.name for p in classified_faulty)
    actual_healthy_names = set(p.name for p in actual_healthy)
    actual_faulty_names = set(p.name for p in actual_faulty)

    # All filenames (should be the same in all sets)
    all_filenames = sorted(classified_healthy_names | classified_faulty_names)

    y_true = []
    y_pred = []

    for fname in all_filenames:
        # True label: 0 for healthy, 1 for faulty
        if fname in actual_faulty_names:
            y_true.append(1)
        else:
            y_true.append(0)
        # Predicted label: 0 for healthy, 1 for faulty
        if fname in classified_faulty_names:
            y_pred.append(1)
        else:
            y_pred.append(0)

    # Compute confusion matrix elements as before
    TP = len(classified_faulty_names & actual_faulty_names)
    FP = len(classified_faulty_names & actual_healthy_names)
    TN = len(classified_healthy_names & actual_healthy_names)
    FN = len(classified_healthy_names & actual_faulty_names)

    return TP, FP, TN, FN, y_true, y_pred

if __name__ == "__main__":
    TP, FP, TN, FN, y_true, y_pred = compute_confusion_matrix_from_folders(
        classified_healthy_path,
        classified_faulty_path,
        actual_healthy_path,
        actual_faulty_path
    )
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

plt.figure(figsize=(8, 8))
cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.savefig(str(project_root / 'plots' /'confusion_matrix_algo.png'))
plt.close()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")