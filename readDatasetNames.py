import os

phase = "train"  # or "train"
root_folder = "./dataset/" + phase + "/"
file_list = os.listdir(root_folder)
instance_names = set([f[:4] for f in file_list if len(f) > 10])

f_original = open("./dataset/" + phase + "_instance_names.txt", "w+")
for name in sorted(list(instance_names)):
    f_original.write(root_folder + name + "\r\n")
f_original.close()

f_train = open("./dataset/" + phase + "_instance_names.txt", "r")
train_data_name = f_train.readlines()
f_train.close()
print(len(train_data_name))
print(len(instance_names))

phase = "test"  # or "train"
root_folder = "./dataset/" + phase + "/"
file_list = os.listdir(root_folder)
instance_names = set([f[:4] for f in file_list if len(f) > 10])

f_original = open("./dataset/" + phase + "_instance_names.txt", "w+")
for name in sorted(list(instance_names)):
    f_original.write(root_folder + name + "\r\n")
f_original.close()

f_train = open("./dataset/" + phase + "_instance_names.txt", "r")
train_data_name = f_train.readlines()
f_train.close()

print(len(train_data_name))
print(len(instance_names))
