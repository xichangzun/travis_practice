
import os
file_path1 = '/data/ABUS_cache/Stomach_cancer/data/train'
file_path2 = '/data/ABUS_cache/Stomach_cancer/data/validate'
train_move = file_path1+'/'+'1'
val_move = file_path2+'/'+'1'

#move all positive-pic under 0.5
for i in os.listdir(val_move):
    name = i.split('_')[-1]
    num = float(name[:3])
    if num < 0.5:
        os.rename((val_move+'/'+i), '/data/ABUS_cache/Stomach_cancer/data_0.5/validate/'+i)

for i in os.listdir(train_move):
    name = i.split('_')[-1]
    num = float(name[:3])
    if num < 0.5:
        os.rename((train_move+'/'+i), '/data/ABUS_cache/Stomach_cancer/data_0.5/train/'+i)