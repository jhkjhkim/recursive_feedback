import numpy as np
import pandas as pd
import sys, pickle
import cv2
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, auc, roc_curve

# https://github.com/qubvel/segmentation_models
import segmentation_models as sm
from segmentation_models.metrics import IOUScore

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import *
import keras.backend as K

import tensorflow as tf
import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # memory increase is needed before starting program
        print(e)


def preprocess_input(X): return X / 127.5 - 1  # to be in range [0,1]


def preprocess_output(Y): return np.asarray(Y > 0.1 * np.max(Y), dtype=np.uint8)


dataset_list = ["kvasir", "cvc", "ISIC2017",  "ETIS-LaribPolypDB", "Shenzhen_Chest_X-ray", "ultrasound_nerve"]
model_list = ["Unet", "Linknet", "FPN", "PSPNet"]

data_id = int(sys.argv[1])  # 0,1,2,3,4 (total 5 dataset)
model_id = int(sys.argv[2])  # 0, 1, 2, 3
seed_id = int(sys.argv[3])  # 10 random seed (0-9)


dataset = dataset_list[data_id]
model_name = model_list[model_id]

image_dim = 384  # to be fixed

with open(dataset + "/" + dataset + "_" + str(image_dim) + '.pickle', 'rb') as f:
    [X, Y] = pickle.load(f)
    print(X.shape, Y.shape)

Y = np.expand_dims(Y, 3)

assert np.max(X) == 255
assert np.min(X) == 0
assert np.max(Y) == 1
assert np.min(Y) == 0

# data split
X_trnval, X_tst, Y_trnval, Y_tst = train_test_split(X, Y, test_size=1.5 / 10, random_state=seed_id)
X_trn, X_val, Y_trn, Y_val = train_test_split(X_trnval, Y_trnval, test_size=1.5 / 8.5, random_state=seed_id)

# testset with target mask only
print("there are {} images in testset".format(len(Y_tst)))
target_index = []
for index, mask in enumerate(Y_tst):
    if mask.sum() > 0:
        target_index.append(index)
    else:
        continue
target_index = np.asarray(target_index)
X_tst = X_tst[target_index]
Y_tst = Y_tst[target_index]
print("there are {} images with target mask in testset".format(len(Y_tst)))

# model construction (keras + sm)
if model_id == 0:
    model = sm.Unet(classes=1, activation='sigmoid', encoder_weights='imagenet')
elif model_id == 1:
    model = sm.Linknet(classes=1, activation='sigmoid', encoder_weights='imagenet')
elif model_id == 2:
    model = sm.FPN(classes=1, activation='sigmoid', encoder_weights='imagenet')
elif model_id == 3:
    model = sm.PSPNet(classes=1, activation='sigmoid', encoder_weights='imagenet')  # input size must be 384x384

data_gen_args = dict(rotation_range=360, width_shift_range=0.15, height_shift_range=0.15, zoom_range=0.15,
                     brightness_range=[0.7, 1.3], horizontal_flip=True, vertical_flip=False, fill_mode='nearest')

image_generator = ImageDataGenerator(**data_gen_args, preprocessing_function=preprocess_input)
mask_generator = ImageDataGenerator(**data_gen_args, preprocessing_function=preprocess_output)

batch_size = 10

image_flow = image_generator.flow(X_trn, seed=seed_id, batch_size=batch_size)
mask_flow = mask_generator.flow(Y_trn, seed=seed_id, batch_size=batch_size)
data_flow = zip(image_flow, mask_flow)

tmpbatchx = image_flow.next()
tmpbatchy = mask_flow.next()
assert np.max(tmpbatchx) == 1
assert np.min(tmpbatchx) == -1
assert np.max(tmpbatchy) == 1
assert np.min(tmpbatchy) == 0

X_trn = preprocess_input(X_trn)
assert np.max(X_trn) == 1
assert np.min(X_trn) == -1

X_val = preprocess_input(X_val)
assert np.max(X_val) == 1
assert np.min(X_val) == -1

X_tst = preprocess_input(X_tst)
assert np.max(X_tst) == 1
assert np.min(X_tst) == -1

print(len(X_trn), len(X_val), len(X_tst))

# prediction
print(':: prediction')
model = load_model(dataset + '/models/' + dataset + '_' + model_name + '_' + "seed_" + str(seed_id) + '.h5',
                   custom_objects={'iou_score': IOUScore(threshold=0.5, per_image=True)})

decay = 0.9
lam = 0.1
T = 5

a = 0
b = 0

single_result_iou = []
ensemble_result_iou = []
single_result_auroc = []
ensemble_result_auroc = []

for it in range(T + 1):

    if it == 0:
        aX_tst = np.copy(X_tst)

    else:
        blur_level = 3
        X_tst_blur = np.array([cv2.blur(x, (blur_level, blur_level)) for x in aX_tst])
        aX_tst = X_tst * lam + (1 - lam) * (aX_tst * Y_tst_score + X_tst_blur * (1 - Y_tst_score))

    Y_tst_score = model.predict(aX_tst, batch_size=10)[:, :, :, :1]
    print(it, np.sum(Y_tst_score > 0.5), np.min(aX_tst), np.max(aX_tst))

    Y_tst_hat = (Y_tst_score > 0.5) + 0
    iou_list = np.array([jaccard_score(Y_tst[i, :, :, 0].ravel(), Y_tst_hat[i].ravel()) for i in range(len(Y_tst))])

    auroc_list = []
    for i in range(len(Y_tst)):
        fpr, tpr, _ = roc_curve(Y_tst[i].ravel(), Y_tst_score[i].ravel())
        roc_auc = auc(fpr, tpr)
        auroc_list.append(roc_auc)
    auroc_list = np.asarray(auroc_list)


    print('iou single at', it, 'th iteration', np.mean(iou_list))
    single_result_iou.append(np.mean(iou_list))
    print('auroc single at', it, 'th iteration', np.mean(auroc_list))
    single_result_auroc.append(np.mean(auroc_list))

    a = a + (decay ** it) * Y_tst_score
    b = b + (decay ** it)

    Y_tst_score = (a / b)
    Y_tst_hat = (a / b > 0.5) + 0
    iou_list = np.array([jaccard_score(Y_tst[i, :, :, 0].ravel(), Y_tst_hat[i].ravel()) for i in range(len(Y_tst))])
    auroc_list = []
    for i in range(len(Y_tst)):
        fpr, tpr, _ = roc_curve(Y_tst[i].ravel(), Y_tst_score[i].ravel())
        roc_auc = auc(fpr, tpr)
        auroc_list.append(roc_auc)
    auroc_list = np.asarray(auroc_list)

    print('iou ensemble at', it, 'th iteration', np.mean(iou_list), b)
    ensemble_result_iou.append(np.mean(iou_list))
    print('auroc ensemble at', it, 'th iteration', np.mean(auroc_list), b)
    ensemble_result_auroc.append(np.mean(auroc_list))


f = open(dataset + '/results/' + dataset + '_' + model_name + '_' + "seed_" + str(seed_id) + "_iou" + '.csv', 'w',
         newline='')
with f:
    writer = csv.writer(f)
    writer.writerow(single_result_iou)
    writer.writerow(ensemble_result_iou)
    writer.writerow(single_result_auroc)
    writer.writerow(ensemble_result_auroc)

if seed_id == 9:  # 9 is maximum random seed
    df = pd.read_csv(dataset + '/results/' + dataset + '_' + model_name + '_' + "seed_" + str(seed_id) + "_iou" + '.csv', header=None)
    rows, cols = df.shape
    result_array = np.empty((rows, cols, seed_id + 1))
    for i in range(seed_id + 1):
        df_2 = pd.read_csv(dataset + '/results/' + dataset + '_' + model_name + '_' + "seed_" + str(i) + "_iou" + '.csv', header=None)
        result_array[:, :, i] = df_2.values

    average = np.mean(result_array, axis=2)
    std = np.std(result_array, axis=2)

    df_avg = pd.DataFrame(average)
    df_std = pd.DataFrame(std)

    df_avg.to_csv(dataset + '/results/' + dataset + '_' + model_name + '_average' + '.csv', header=None, index=False)
    df_std.to_csv(dataset + '/results/' + dataset + '_' + model_name + '_std' + '.csv', header=None, index=False)