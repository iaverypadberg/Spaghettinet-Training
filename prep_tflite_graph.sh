CONFIG_FILE="/home/da/Desktop/test_tflite_export/pipeline.config"
CHECKPOINT_PATH="/home/da/Desktop/test_tflite_export/model.chkpt-400000"
OUTPUT_DIR="/home/da/Desktop/test_tflite_export/output"
python3 /path/to/models/reasearch/object_detection/export_tflite_ssd_graph.py \
	--pipeline_config_path=${CONFIG_FILE} \
	--trained_checkpoint_prefix=${CHECKPOINT_PATH} \
	--output_directory=${OUTPUT_DIR} \
	--add_postprocessing_op=true
