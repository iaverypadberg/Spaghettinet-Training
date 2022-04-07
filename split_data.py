import os
import random
import subprocess
input_dir ='your/dataset/here'
out_train ='where/you/want/to/store/train'
out_test ='where/you/want/to/store/test'
count=0
for file in os.listdir(input_dir):

    file_check = os.path.splitext(file)[1]
    if file_check == ".png":
        rand=random.randint(0,100)
        # 25% of the data is in test
        if(rand>75):
            os.system('cp {}/{} {}'.format(input_dir,file, out_test))
            xml_file = os.path.splitext(file)[0]
            os.system('cp {}/{}{} {}'.format(input_dir,xml_file,".xml",out_test))
 
        # 75% of the data is in train
        if(rand<=75):
            os.system('cp {}/{} {}'.format(input_dir,file,out_train))
            xml_file = os.path.splitext(file)[0]
            os.system('cp {}/{}{} {}'.format(input_dir,xml_file,".xml",out_train))
    count+=1

print("The total number of files copied:{}".format(count))

