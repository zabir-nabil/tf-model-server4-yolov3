import tensorflow as tf
import os

# change these 3 values according to your need

ckpt_dir = 'E:/yolopb_model_test/tf_model_server_test/training'
ckpt_file = 'yolov3-lp_vehicles.ckpt.meta'

# find the output node by using netron or look into the graph.log file from this script

input_node_names = ['yolov3-lp_vehicles_v1/net1']
output_node_names = ['yolov3-lp_vehicles_v1/convolutional59/BiasAdd']    # Output nodes

export_path = 'exported_model'

meta_path = os.path.join(ckpt_dir, ckpt_file) # Your .meta file


graph_log = open('graph.log', 'a+')

with tf.Session() as sess:
    
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
    
    graph_ops = tf.get_default_graph().get_operations()
    
    for op in graph_ops:
        graph_log.write(str(op) + '\n')
            
    in_ = tf.get_default_graph().get_tensor_by_name(input_node_names[0] + ':0')
    out_ = tf.get_default_graph().get_tensor_by_name(output_node_names[0] + ':0')
    
    print(in_)
    print(out_)
    
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # input tensor info
    tensor_info_input = tf.saved_model.utils.build_tensor_info(in_)
    # output tensor info
    tensor_info_output = tf.saved_model.utils.build_tensor_info(out_)

    # Defines the DeepLab signatures, uses the TF Predict API
    # It receives an image and its dimensions and output the segmentation mask
    
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'image': tensor_info_input},
            outputs={'yolo_out': tensor_info_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
                'predict_images':
                prediction_signature,
        })

    # export the model
    builder.save(as_text=True)
    print('Done exporting!')
    