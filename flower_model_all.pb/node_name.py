import tensorflow.compat.v1 as tf
import tensorflow as tf
import tf2onnx

tf.disable_eager_execution() 

# Load the frozen graph (assuming it's a tf.GraphDef)
with tf.io.gfile.GFile("/media/ResNet/flower_class/flower_model_all.pb/saved_model.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Define input and output node names
input_names = ["input_node_name:0"]
output_names = ["output_node_name:0"]

# Convert the graph to ONNX
onnx_graph = tf2onnx.tfonnx.process_tf_graph(graph_def, input_names=input_names, output_names=output_names)

# Save the ONNX model
model_proto = onnx_graph.make_model("your_model_name")
with open("your_model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
