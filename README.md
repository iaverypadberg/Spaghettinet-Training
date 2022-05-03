# Training a SpaghettiNet Model

## Dataset

### Dataset format

* Each image should be annotated and have a corresponding .xml file
* *All* the images and their corresponding xml files shouls be located in the same folder.

``` text

├── generate_tfrecord.py
├── all_images_here
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

---

#### Note

If any of the files fail to copy, then that file does not have a matching .xml file. You should remove the corresponding .png or .jpg file from your dataset and and the newly create data split.

---

### XML to CSV

* As long as your folder structure and names are the same as mine, you should not have any issues with the below comand to generate the .csv for both train and test.
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

Gotta have a label map.
Make the label_map.txt file in the training folder and populate it with the following format.

``` Text
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

Arriving at this configuration took some time, but in the end it was pretty simple. I am training on a GPU, hence I install tensorflow-GPU instead of just plain tensorflow=1.15

You will first need to installl miniconda/conda. Then run this command:

``` python
conda create --name tensorflow-15 \
    tensorflow-gpu=1.15 \
    cudatoolkit=10.0 \
    cudnn=7.6 \
    python=3.6 \
    pip=20.0
```

For training on CPU

``` python
conda create --name tensorflow-15 \
    tensorflow=1.15 \
    python=3.6 \
    pip=20.0
```

This creates a virtual environment where the training will be done.

You should be set to train after running the above command, but nin the case that there are missing libraries just install them with pip.

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

``` Text

train_input_reader: {
  label_map_path: "/path/to/label/map/label_map.txt"
  tf_record_input_reader {
    input_path: "/path/to/train/record/train.record"
  }
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
}

eval_input_reader: {
  label_map_path: "/path/to/label/map/label_map.txt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "/path/to/test/record/test.record"
  }
}

```

### Quantization

At the bottom of the pipeline.config file there is a section that looks like this:
``` text
graph_rewriter {
  quantization {
    delay: 40000
    weight_bits: 8
    activation_bits: 8
  }
}
```

This graph_rewriter flag lets tensorflow know that after the number of steps is greater than the `delay`, that it should train with quantization aware training. The delay [should be set at](https://discuss.tensorflow.org/t/training-a-spaghettinet-model/8648/15?u=isaac_padberg) about 10% of the total training steps.

If you want this model to execute on an edge device I recommend you quantize your model :)


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

There are lots of possible errors when training, but I ran into [Data Loss](https://github.com/tensorflow/tensorflow/issues/13463#issuecomment-381818710) error, and solved it by using this command to clear the momory caches. 

``` 
sudo sh -c "sync; echo 1 > /proc/sys/vm/drop_caches"
```

I also ran into a fair amount of "OUT OF MEMORY" errors. In the standard pipeline.config file that is provided by Tensorflow, the batch_size is set to 512. A singular 12GB GTX 2080Ti can only handle a max batch size of 16, so scale accordingly.

## Convert to tflite

### Convert checkpoints of a trained model to a tflite_graph.pb

---

#### NOTE

Tensorflow checkpionts are made up of all three files(data,meta,index), to specify a checkpoint you only need to pass in the prefix. The prefix for a particular checkpioint usually looks like this: model.ckpt-200

---

``` python

python object_detection/export_tflite_ssd_graph.py 
--pipeline_config_path=$CONFIG_FILE 
--trained_checkpoint_prefix=$CHECKPOINT_PATH 
--output_directory=$OUTPUT_DIR 
--add_postprocessing_op=true
```

This should give you the these two files in the output directory you specified.

``` text
├── tflite_graph.pb
└── tflite_graph.pbtxt

```

You can now use the tflite_graph.pb as input into the next command.

### Export as a tflite model

* This is the default implementation of the SpaghettiNet Model shown in their Tensorflows model garden and in TFHub. 
* Typing this into terminal is a bit annoying, so I've included a export_tflite.sh script to run the command. Modify the path at the top.

--

### NOTE

 **default_ranges_min/max are set to 0 and 6 because the operation that uses these defaults is the Relu_6 op. You can play around with the numbers, but it might lead to some pretty poor accuracy.**

---

``` python

This converts a tflite ready ssd model to a real tflite model
tflite_convert --graph_def_file=$OUTPUT_DIR/tflite_graph.pb 
--output_file=$OUTPUT_DIR/spaghetti.tflite 
--input_shapes=1,320,320,3 
--input_arrays=normalized_input_image_tensor 
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' 
--inference_type=QUANTIZED_UINT8 
--mean_values=128 
--std_dev_values=128 
--change_concat_input_ranges=false 
--allow_custom_ops 
#--default_ranges_min=0 
#--default_ranges_max=6
```

### Add metadata to the model

* To have this model run on the tflite interpreter, it needs some metadata added to it.
* The metadata includes labels, quantization, and othe input settings.
* I executed this script in a TF 2.8.0 environment, and it works there. Im not sure about TF 1.x though.
* The mean and std params are specified in this metadata. Im not sure what they should be set to for this SpaghettiNet model because this model has float output and uint8 input tensors.

Change the file paths at the top of the script to match your file locations and names.

``` python

python3 add_metadata.py

```
