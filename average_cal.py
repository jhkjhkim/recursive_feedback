import pandas as pd
import numpy as np
import sys, csv

dataset_list = ["kvasir", "cvc", "ISIC2017", "ultrasound_nerve", "ETIS-LaribPolypDB"]
model_list = ["Unet", "Linknet", "FPN", "PSPNet"]

data_id = int(sys.argv[1])  # 0,1,2,3,4 (total 5 dataset)
model_id = int(sys.argv[2])  # 0, 1, 2, 3
#seed_id = int(sys.argv[3])  # 10 random seed (0-9)
seed_id = 9

dataset = dataset_list[data_id]
model_name = model_list[model_id]

image_dim = 384  # to be fixed

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

    df_avg.to_csv(dataset + '/results/' + dataset + '_' + model_name + '_average_iou' + '.csv', header=None, index=False)
    df_std.to_csv(dataset + '/results/' + dataset + '_' + model_name + '_std_iou' + '.csv', header=None, index=False)