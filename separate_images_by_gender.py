#separate UTKFace images by gender
from shutil import move
import os
import csv

root_dir = "UTKFace"

def separate_into_train_and_test():
    train_male_dir = os.path.join(root_dir, 'train', 'male')
    train_female_dir = os.path.join(root_dir, 'train', 'female')
    test_male_dir = os.path.join(root_dir, 'test', 'male')
    test_female_dir = os.path.join(root_dir, 'test', 'female')
    if not os.path.exists(train_male_dir):
        os.makedirs(train_male_dir)
    if not os.path.exists(train_female_dir):
        os.makedirs(train_female_dir)
    if not os.path.exists(test_male_dir):
        os.makedirs(test_male_dir)
    if not os.path.exists(test_female_dir):
        os.makedirs(test_female_dir)

    female_dir = os.path.join('UTKFace', 'female')
    male_dir = os.path.join('UTKFace', 'male')

    index = 0
    for filename in os.listdir(female_dir):
        components = filename.split("_")
        #Assume file names are correctly formatted, with gender label at index 1
        if(len(components) <2): continue
        if index % 10 == 0:
            move(os.path.join(female_dir,filename), os.path.join(test_female_dir, filename))
        else:
            move(os.path.join(female_dir,filename), os.path.join(train_female_dir, filename))
        index+= 1

    index = 0
    for filename in os.listdir(male_dir):
        components = filename.split("_")
        #Assume file names are correctly formatted, with gender label at index 1
        if(len(components) <2): continue
        if index % 10 == 0:
            move(os.path.join(male_dir,filename), os.path.join(test_male_dir, filename))
        else:
            move(os.path.join(male_dir,filename), os.path.join(train_male_dir, filename))
        index+= 1

def collect_filenames_in_csv(image_dir_path, csv_path):
    count = 0
    with open(csv_path, 'w') as csvfile:
                fieldnames = ["image_name"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for filename in os.listdir(image_dir_path):
                    count += 1
                    writer.writerow({"image_name": filename})
    print("total numer of files: {}".format(count))

collect_filenames_in_csv("UTKFace/female", "utkface_all_female.csv")
collect_filenames_in_csv("UTKFace/male", "utkface_all_male.csv")
