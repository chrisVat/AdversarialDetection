import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import os 
import pickle
from tqdm import tqdm
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope


print("Converting to numpy arrays...")
clean_data_train = []
adversarial_data_train = []
clean_data_test = []
adversarial_data_test = []

def get_data(dataset_name):
    if not os.path.exists("adv_datasets/OUTLIER_DATASETS/" + dataset_name + "/clean_data_train.npy"):
        train_mapping = pd.read_csv("adv_datasets/" + dataset_name + "/" + dataset_name + "_train/mapping.csv")  
        test_mapping = pd.read_csv("adv_datasets/" + dataset_name + "/" +dataset_name + "_test/mapping.csv")

        # Load numpy files into clean_data_train and adversarial_data_train using with tqdm
        for index, row in tqdm(train_mapping.iterrows(), total=len(train_mapping)):
            if row["y"] == 0:
                clean_data_train.append(np.load("adv_datasets/" +dataset_name + "/" + dataset_name+"_train/" + row["file"]))
            elif row["y"] == 1:
                adversarial_data_train.append(np.load("adv_datasets/" + dataset_name + "/" +dataset_name + "_train/" + row["file"]))

        for index, row in tqdm(test_mapping.iterrows(), total=len(test_mapping)):
            if row["y"] == 0:
                clean_data_test.append(np.load("adv_datasets/" +dataset_name + "/" + dataset_name+"_test/" + row["file"]))
            elif row["y"] == 1:
                adversarial_data_test.append(np.load("adv_datasets/" +dataset_name + "/" + dataset_name+"_test/" + row["file"]))        

        clean_data_train = np.array(clean_data_train)
        adversarial_data_train = np.array(adversarial_data_train)
        clean_data_test = np.array(clean_data_test)
        adversarial_data_test = np.array(adversarial_data_test)

        clean_data_train = clean_data_train.reshape(-1, clean_data_train.shape[-1])
        adversarial_data_train = adversarial_data_train.reshape(-1, adversarial_data_train.shape[-1])
        clean_data_test = clean_data_test.reshape(-1, clean_data_test.shape[-1])
        adversarial_data_test = adversarial_data_test.reshape(-1, adversarial_data_test.shape[-1])


        clean_data_train.dump("adv_datasets/OUTLIER_DATASETS/" + dataset_name + "/clean_data_train.npy")
        adversarial_data_train.dump("adv_datasets/OUTLIER_DATASETS/" + dataset_name + "/adversarial_data_train.npy")
        clean_data_test.dump("adv_datasets/OUTLIER_DATASETS/" + dataset_name + "/clean_data_test.npy")
        adversarial_data_test.dump("adv_datasets/OUTLIER_DATASETS/" + dataset_name + "/adversarial_data_test.npy")

    else:
        print("Skipping mapping...")    
        clean_data_train = np.load("adv_datasets/OUTLIER_DATASETS/" + dataset_name + "/clean_data_train.npy", allow_pickle=True)
        adversarial_data_train = np.load("adv_datasets/OUTLIER_DATASETS/" + dataset_name + "/adversarial_data_train.npy", allow_pickle=True)
        clean_data_test = np.load("adv_datasets/OUTLIER_DATASETS/" + dataset_name + "/clean_data_test.npy", allow_pickle=True)
        adversarial_data_test = np.load("adv_datasets/OUTLIER_DATASETS/" + dataset_name + "/adversarial_data_test.npy", allow_pickle=True)
        return clean_data_train, adversarial_data_train, clean_data_test, adversarial_data_test


def test_method_solo(method, clean_data_train, adversarial_data_train, clean_data_test, adversarial_data_test, name):
    # LocalOutlierFactor
    print("Training ", name, " Classifier...")
    method.fit(clean_data_train)
    clean_preds = method.predict(clean_data_test)
    adv_preds = method.predict(adversarial_data_test)*-1

    # set values less than 0 to 0
    print("Calculating accuracy...")
    clean_preds[clean_preds < 0] = 0
    adv_preds[adv_preds < 0] = 0
    print(name + " Clean Accuracy: ", np.sum(clean_preds)/len(clean_preds))
    print(name + " Adversarial Accuracy: ", np.sum(adv_preds)/len(adv_preds))
    print(name + " Total Accuracy: ", (np.sum(clean_preds) + np.sum(adv_preds))/(len(clean_preds) + len(adv_preds)))

    print("------------------------------------")

    print("Training Adversarial " + name + " Classifier...")
    method.fit(adversarial_data_train)
    clean_preds = method.predict(clean_data_test)*-1
    adv_preds = method.predict(adversarial_data_test)

    # set values less than 0 to 0
    print("Calculating accuracy...")
    clean_preds[clean_preds < 0] = 0
    adv_preds[adv_preds < 0] = 0
    print(name + " Clean Accuracy: ", np.sum(clean_preds)/len(clean_preds))
    print(name + " Adversarial Accuracy: ", np.sum(adv_preds)/len(adv_preds))
    print(name + " Total Accuracy: ", (np.sum(clean_preds) + np.sum(adv_preds))/(len(clean_preds) + len(adv_preds)))


def test_method_combined(method, clean_data_train, adversarial_data_train, clean_data_test, adversarial_data_test, name):
    combined_data = np.concatenate((clean_data_train, adversarial_data_train))
    print("Training Combined", name, " Classifier...")
    method.fit(combined_data)
    clean_preds = method.predict(clean_data_test)
    adv_preds = method.predict(adversarial_data_test)*-1

    clean_preds[clean_preds < 0] = 0
    adv_preds[adv_preds < 0] = 0
    print(name + "Clean Accuracy: ", np.sum(clean_preds)/len(clean_preds))
    print(name + "Adversarial Accuracy: ", np.sum(adv_preds)/len(adv_preds))
    print(name + "Total Accuracy: ", (np.sum(clean_preds) + np.sum(adv_preds))/(len(clean_preds) + len(adv_preds)))



dataset = "ciless_embeddings_vit_pretrain"

clean_data_train, adversarial_data_train, clean_data_test, adversarial_data_test = get_data(dataset)
solo_methods = {
    "IsolationForest": IsolationForest(contamination=0.0001, random_state=0),
    "EllipticEnvelope": EllipticEnvelope(contamination=0.0001)
}

combined_methods = {
    "IsolationForest": IsolationForest(contamination=0.5, random_state=0),
    "EllipticEnvelope": EllipticEnvelope(contamination=0.5)
}

for method in solo_methods:
    meth = solo_methods[method]
    print("\n\n")
    test_method_solo(meth, clean_data_train, adversarial_data_train, clean_data_test, adversarial_data_test, method)
    print("\n")
    test_method_combined(combined_methods[method], clean_data_train, adversarial_data_train, clean_data_test, adversarial_data_test, method)

