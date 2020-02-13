# tf-model-server4-yolov3
Simple code base and instructions to convert yolov3 darknet weights to tensorflow .pb to serve @ tensorflow model server

### Convert .weights to .ckpt

Use DW2TF repository

https://github.com/jinyu121/DW2TF


### Deploy to tf-model-server

> Go to export.py

> Change the path and filename for .ckpt files

> Change the input, output nodes for yolo (you can find it using Netron or the graph.log file which will be generated after running export.py [even if it fails, the graph.log will be generated])

> command

`python export.py`

> The model will be found in exported_model folder

> Copy the .pb / .pbtxt and the variables folder to serving path (~serving/versions/)

##### Run gRPC server

`tensorflow_model_server --port=9000 --model_name=yolo --model_base_path=/absolute_path_to/yolo_v3/serving/versions/`


##### Run REST Server

> To serve with GPU
```
nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=yolo \
  --model_base_path=/absolute_path_to/yolo_v3/serving/versions/
  -t tensorflow/serving:gpu >server.log 2>&1
```

> To serve at CPU
```
nohup tensorflow_model_server \
  --rest_api_port=8502 \
  --model_name=yolo \
  --model_base_path=/absolute_path_to/yolo_v3/serving/versions/
  -t tensorflow/serving:latest >server.log 2>&1
```

##### Docker/ GPU

> Docker KILL

`docker container ls`

`docker stop ID`

> Clear GPU memory

`nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9`

##### Inference / API call

> Look into the * test_api.py * for both gRPC and REST


### Flask Server for native darknet YOLOv3

> A light flask server for darknet yolov3 is in darknet_server folder

> Run darknet_server.py

> Has both numpy image and base64 image support

> It is slightly faster than tensorflow-model-server based on some benchmarks, but it has very limited functionalities

> Not scalable yet
