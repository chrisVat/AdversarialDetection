from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from custom_adversarial_dataset import AdversarialDataset

def evaluate_detection(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print("True Negative Rate: ", TN)
    print("False Negative Rate: ", FN)
    print("True Positive Rate: ", TP)
    print("False Positive Rate: ", FP)

    score = precision_score(y_true, y_pred)
    print("Precision Score for detection: ", score)

def extract_adversarial_attack_labels(dataset: AdversarialDataset):
    attack_labels = []
    for sample in dataset:
        attack_labels.append(sample["is_attacked"])

    return attack_labels

def test_detector(dataset: AdversarialDataset, y_pred):
    attack_labels = extract_adversarial_attack_labels(dataset)
    evaluate_detection(attack_labels, y_pred)

