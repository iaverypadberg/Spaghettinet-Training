
OUT_DIR=/home/da/Desktop/spag-train1
tflite_convert --output_file=${OUT_DIR}/spaghetti.tflite \
	--graph_def_file=${OUT_DIR}/tflite_graph.pb \
	--input_shapes=1,320,320,3 \
	--input_arrays=normalized_input_image_tensor \
	--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
	--inference_type=QUANTIZED_UINT8 \
	--mean_values=128 \
	--std_dev_values=128 \
	--change_concat_input_ranges=false \
	--allow_custom_ops \
	--default_ranges_min=-128 \
	--default_ranges_max=128 
