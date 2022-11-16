import os
import time
from main_generate_sag import main

folder = r'vgg19_20220910_201515'
subfolder = 'vgg19_20220910_201515_epoch_9'

ls = [8,11,12]
model_name = folder.split("_")[0]

root = r'/media/kv/Documents/git/mtkvcs-saved-models/'
dataset_path = r'/media/kv/Documents/git/mtkvcs-custom-cnn/dataset/datscan/'
unique_id = time.strftime("%Y%m%d_%H%M%S")
results_path = root + r'/' + folder + r'/Results/'+ unique_id

loop=0
if loop==0:
    for j in os.listdir(root + folder + r'/' + subfolder):
            if ".pt" in j:
                print("Run only model: " + j)
                main(root + folder + r'/' + subfolder + r'/' + j, folder, results_path)
elif loop==1:
    for i in os.listdir(root+folder):
        if "epoch" in i:
            for j in os.listdir(root+folder + r'/' +i):
                if ".pt" in j:
                    main(root + folder + r'/' + i + r'/' + j, folder, results_path)
elif loop == -1:
    print("Old format")
    main(root + folder + r'/' + folder + r'.pt', folder, results_path)