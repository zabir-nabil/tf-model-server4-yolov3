# tf-model-server4-yolov3
Simple code base and instructions to convert yolov3 darknet weights to tensorflow .pb to serve @ tensorflow model server

# Convert .weights to .ckpt

Use DW2TF repository

https://github.com/jinyu121/DW2TF


# Deploy to tf-model-server

> Go to export.py

> Change the path and filename for .ckpt files

> Change the input, output nodes for yolo (you can find it using Netron or the graph.log file which will be generated after running export.py [even if it fails, the graph.log will be generated])

> command

`python export.py`

> The model will be found in exported_model folder
