import csv
import sys
import os

def split_names_to_csv(path_name, out_path):
    absolute_path = os.path.join(os.getcwd(), path_name)
    if(os.path.isdir(absolute_path)):
        img_names = [fl.split("_") for fl in os.listdir(absolute_path) if os.path.isfile(os.path.join(absolute_path, fl))]               
        absolute_output_path = os.path.join(os.getcwd(), out_path)
        exists = os.path.isfile(absolute_output_path)
        with open(absolute_output_path, "a") as output:
            write_file = csv.writer(output)
            if(not exists):
                write_file.writerow(["age", "gender", "race", "image name"])
            write_file.writerows(img_names)
        output.close()
        
    else:
        print("directory %s does not exist" %(absolute_path))
        exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3 :
        print("Format: python create_csv.py path_to_csv output_name")
        exit(1)
        
    path_name = sys.argv[1] #should be a path from the current working directory to the train samples with labels age_gender_race_img
    out_path = sys.argv[2] #name of output csv for comma separated csv into columns for race, gender, age and image label
    split_names_to_csv(path_name, out_path)
     
    