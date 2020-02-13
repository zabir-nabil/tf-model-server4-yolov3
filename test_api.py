# python3

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import requests
import numpy as np
from io import StringIO, BytesIO
import cv2
import json

# gRPC demo

# define your host, port
server = '192.168.106.222:8500'
host, port = server.split(':')

# give the path to test image
image_path = 'test.png'

# assuming yolo input shape is (416,416)
image = cv2.resize(cv2.imread(image_path), (416,416)) # may have to normalize

print(f"Image shape: {image.shape[0]}, {image.shape[1]}")

channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = predict_pb2.PredictRequest()

# may need to change if you changed these while exporting
request.model_spec.name = 'yolo' 
request.model_spec.signature_name = 'predict_images'

request.inputs['image'].CopyFrom(
  tf.contrib.util.make_tensor_proto(image.astype(dtype=np.float32), shape=[1, 416, 416, 3]))

r = stub.Predict(request, 3.)

print(r)


# REST demo



payload = json.dumps({'name': 'yolo', 'signature_name' : 'predict_images',
                     'inputs' : {'image': image.tolist()}})

r = requests.post("http://" + host + ":" + port + "/v1/models/yolo:predict", data = payload)

print(r.json())
