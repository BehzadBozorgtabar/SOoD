import os
import numpy as np
import json


def split(data, pct_train, pct_test, pct_data=1.0):
    """This method splits the given data into a training, a validation and a testing set. It shuffles data before
    the split.

    Args:
        data ([]): the list of data
        pct_train (float): the percentage of data which should be in the train set
        pct_test (float): the percentage of data which should be in test set
        pcd_data (float): the percentage of data to load in the sets. Defaults to 1.0.

    Returns:
        ([], [], []): the training, the validation and the test datasets
    """

    total_counts = len(data)
    print("There are {:d} samples".format(total_counts))
    indexes = np.arange(total_counts)
    np.random.seed(15)
    np.random.shuffle(indexes)

    train_indexes = indexes[:int(pct_train * total_counts)]
    train_indexes = train_indexes[:int(pct_data * len(train_indexes))]
    val_indexes = indexes[int(pct_train * total_counts):int((1-pct_test) * total_counts)]
    val_indexes = val_indexes[:int(pct_data * len(val_indexes))]
    test_indexes = indexes[int((1-pct_test) * total_counts):]
    test_indexes = test_indexes[:int(pct_data * len(test_indexes))]

    train_images = np.array(data)[train_indexes].tolist()
    val_images = np.array(data)[val_indexes].tolist()
    test_images = np.array(data)[test_indexes].tolist()

    return train_images, val_images, test_images


#Set the path of the datasets and parameters
kather16_src = "Kather_2016/Images/"
kather19_src = "Kather_2019/Images/"
test_pct = 0.1
train_pct = 0.8
pct_data_19_data = 1.0
anomalies = ["COMPLEX"]

#We list the different folders
kather16_folders = os.listdir(kather16_src)
kather19_folders = os.listdir(kather19_src)

#We initialize the datasets which we will save as json dictionnary
kather16_train = {}
kather16_val = {}
kather16_test = {}
kather19_train = {}
kather19_val = {}
kather19_test = {}

#Set labels to the datasets
kather16_labels = [x.split("_")[-1] for x in kather16_folders]
kather19_labels = [x for x in kather16_folders]

kather16_train["brut_labels"] = kather16_labels
kather16_train["labels"] = kather16_labels
kather16_val["brut_labels"] = kather16_labels
kather16_val["labels"] = kather16_labels
kather16_test["brut_labels"] = kather16_labels
kather16_test["labels"] = kather16_labels

kather19_train["brut_labels"] = kather19_labels
kather19_train["labels"] = kather16_labels
kather19_val["brut_labels"] = kather19_labels
kather19_val["labels"] = kather16_labels
kather19_test["brut_labels"] = kather19_labels
kather19_test["labels"] = kather16_labels

#Set labels mapping
map_labels = {"TUM": "TUMOR", "STR": "STROMA", "MUS": "STROMA", "LYM": "LYMPHO",
           "NORM": "MUCOSA", "DEB": "DEBRIS", "MUC": "DEBRIS", "ADI": "ADIPOSE",
           "BACK": "EMPTY"}

#Split kather16 into train and test set and put in data dictionnary and then save the json file
kather16_train["images"], kather16_val["images"], kather16_test["images"] = [], [], []
for folder in kather16_folders:
    folder_path = os.path.join(kather16_src, folder)
    folder_images = []
    label = folder.split("_")[-1]
    for image in os.listdir(folder_path):
        folder_images.append({"path":os.path.join(folder_path,image),
                      "label": label,
                      "brut_label": label})
    # For OoDs, we keep 70% of the data in the test set, the remaining is going to the validation set
    if label in anomalies:
        train, val, test = split(folder_images, 0.0, 0.7)
    else:
        train, val, test = split(folder_images, train_pct, test_pct)
    kather16_train["images"] += train
    kather16_val["images"] += val
    kather16_test["images"] += test

with open(os.path.join(kather16_src.split("/")[0],"train.json"),"w") as f:
    json.dump(kather16_train,f)
with open(os.path.join(kather16_src.split("/")[0],"val.json"),"w") as f:
    json.dump(kather16_val,f)
with open(os.path.join(kather16_src.split("/")[0],"test.json"),"w") as f:
    json.dump(kather16_test, f)


#We do the same for kather19 dataset, we care to correctly map the labels
kather19_train["images"], kather19_val["images"], kather19_test["images"] = [], [], []
for folder in kather19_folders:
    folder_path = os.path.join(kather19_src, folder)
    folder_images = []
    label = map_labels[folder]
    for image in os.listdir(folder_path):
        folder_images.append({"path":os.path.join(folder_path,image),
                      "label":label,
                      "brut_label":folder})
    # For OoDs, we keep 70% of the data in the test set, the remaining is going to the validation set
    if label in anomalies:
        train, val, test = split(folder_images, 0.0, 0.7)
        kather19_train["images"] += train
        kather19_test["images"] += test
    else:
        train, val, test = split(folder_images, train_pct, test_pct, pct_data_19_data)
    kather19_train["images"] += train
    kather19_val["images"] += val
    kather19_test["images"] += test

with open(os.path.join(kather19_src.split("/")[0],"train.json"),"w") as f:
    json.dump(kather19_train,f)
with open(os.path.join(kather19_src.split("/")[0],"val.json"),"w") as f:
    json.dump(kather19_val,f)
with open(os.path.join(kather19_src.split("/")[0],"test.json"),"w") as f:
    json.dump(kather19_test, f)
