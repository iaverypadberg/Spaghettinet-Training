# SpaghettiNet Training Walkthrough

## Dataset

### Dataset format

* Each image should be annotated and have a corresponding .xml file
* *All* the images and their corresponding xml files shouls be located in the same folder.

``` text

├── generate_tfrecord.py
├── images
│   ├── test
│   └── train
├── split_data.py
├── training
│   ├── label_map.txt
│   ├── pipeline.config
│   ├── training-output
│   └──
└── xml_to_csv.py
```

### Split Data

* Modify the input and output directories at the top of the split_data.py script.
* If you want to change the train test split, you can modify two if statements checking the randint.
* Run the script. `python3 split_data.py`

***Note, if any of the files fail to copy, then that file does not have a matching .xml file. You should remove the corresponding .png or .jpg file from your dataset and and the newly create data split.

### XML to CSV

* As long as your folder strucutre and names are the same as mine, you should not gave any issues with the below comand to generate the .csv for both train and test.
* `python3 xml_to_csv.py`
*** This command generates 2 .csv files.

``` text

images
   ├── test
   ├── test_labels.csv
   ├── train_labels.csv
   └── train

```

### Tfrecord

* To get a tfrecord file that fits you data, you need to edit the class_text_to_int function and add in your classes in the chained 'elif' statements.
* Run the below command for the both the test and train csv labels as you will need a .record for each.

``` python
python3 generate_tfrecord.py --csv_input=images/test_labels.csv --output_path=training/test.record --image_dir=images/test/

```
Resulting file structure

``` text

training
   ├── label_map.txt
   ├── test.record
   └── train.record

```
### Label map

Gotta have a ground truth label map.
Make a file label_map.txt in the training folder and populate it with the following format.

```
item{
        name:"frodo"
        id:1
}
item{
        name:"baggins"
        id:2
}

```

## Environment Configuration

Arriving at this configuration took some time, but in the end it was pretty simple.

You will first need to installl miniconda/conda. Then run this command:

``` python
conda create --name tensorflow-15 \
    tensorflow-gpu=1.15 \
    cudatoolkit=10.0 \
    cudnn=7.6 \
    python=3.6 \
    pip=20.0
```

This creates a virtual environment where the training will be done.

## Model Configuration

The pipleline.config file is the missing link in this whole thing. You can aquire a SpaghettiNet config [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#pixel-6-edge-tpu-models). You will need to download, extract, then find the ```pipeline.config``` file. Move this file into the training directory.

``` text
training
   ├── label_map.txt
   ├── pipeline.config
   ├── test.record
   ├── trained-output
   └── train.record
```

**Also make a trained-output directory to store the training output.**

Once you have the config file in the correct place, you can go ahead and modify paths at the bottom of the file. It should look something like this in the end:

```

train_input_reader: {
  label_map_path: "/home/da/Desktop/spaghetti_train/training/label_map.txt"
  tf_record_input_reader {
    input_path: "/home/da/Desktop/spaghetti_train/training/train.record"
  }
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
}

eval_input_reader: {
  label_map_path: "/home/da/Desktop/spaghetti_train/training/label_map.txt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "/home/da/Desktop/spaghetti_train/training/test.record"
  }
}

```

Modify the hyperparameters as you please.

## Training

To start training, you will first need to install the TF1 object detection API

``` python
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf1/setup.py .
python -m pip install .

```

Check the install with
```
python object_detection/builders/model_builder_tf1_test.py
```

Now you're ready to train!
``` python
python /home/da/Documents/git-repos/models/research/object_detection/model_main.py --model_dir=/home/da/Desktop/spaghetti_train/training/training-output/ --pipeline_config_path=/home/da/Desktop/spaghetti_train/training/pipeline.config

```

## Convert to tflite

TODO
