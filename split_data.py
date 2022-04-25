import os
import random
import subprocess
input_dir ='your/dataset/here' # Contains both xml and jpg,png,jpeg files
out_train ='where/you/want/to/store/train'
out_test ='where/you/want/to/store/test'
count=0
for file in os.listdir(input_dir):
    
    file_check = os.path.splitext(file)[-1]
    #Add '' around the filename to deal with files that have spaces in their names
    modified_file="'"+file+"'"
    if file_check != ".xml":
        rand=random.randint(0,100)
        # 10% of the data is in test
        if(rand>90):
            os.system('cp {}/{} {}'.format(input_dir,modified_file,out_test))
            xml_file = modified_file.split(".")[0]+".xml'"

            os.system('cp {}/{} {}'.format(input_dir,xml_file,out_test))

        # 90% of the data is in train
        if(rand<=90):
            os.system('cp {}/{} {}'.format(input_dir,modified_file,out_train))

            xml_file = modified_file.split(".")[0]+".xml'"
            os.system('cp {}/{} {}'.format(input_dir,xml_file,out_train))

    count+=1

print("The total number of files copied:{}".format(count))

