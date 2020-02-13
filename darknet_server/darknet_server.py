from flask import Flask
from flask_restful import Resource, Api, reqparse
import werkzeug, os
import json
import numpy as np
from darknet import *
import io
from io import BytesIO
from PIL import Image
import cv2
import base64

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



app = Flask(__name__)
api = Api(app)
UPLOAD_FOLDER = 'imgs/'
parser = reqparse.RequestParser()
parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
parser.add_argument('imgb64')

# loading darknet
net = load_net(b"cfg/yolov3-lp_vehicles.cfg", b"backup/yolov3-lp_vehicles.backup", 0)
meta = load_meta(b"data/lp_vehicles.data")

class Test(Resource):
    def get(self):
        return {'status': 'ok'}


class Predict(Resource):
    decorators=[]

    def post(self):
        data = parser.parse_args()
        if data['file'] == "":
            return {
                    'data':'',
                    'message':'No file found',
                    'status':'error'
                    }
        img = data['file'].read()

        print(type(img))

        if img:
            filename = 'img_now.jpg'
            with open(filename, 'wb') as f:
                f.write(img)

            img = cv2.imread('img_now.jpg')

            r = detect_np(net, meta, img)
            print(r)

            #open(filename, 'wb').write(pdffile)

            return json.dumps({
                    'data': str(r), #(images),
                    'message':'pdf uploaded',
                    'status':'success'
                    }, cls=NumpyEncoder)
        return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
                }


class PredictB64(Resource):
    decorators=[]

    def post(self):
        data = parser.parse_args()
        if data['imgb64'] == "":
            return {
                    'data':'',
                    'message':'No file found',
                    'status':'error'
                    }
        img = data['imgb64']
        #print(img)


        nparr = np.fromstring(base64.b64decode(img), np.uint8)
        im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        print(type(im))




        if img:
            r = detect_np(net, meta, im)
            #print(r)

            return json.dumps({
                    'data': str(r), #(images),
                    'message':'darknet processed',
                    'status':'success'
                    }, cls=NumpyEncoder)
        return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
                }

api.add_resource(Test, '/test')
api.add_resource(Predict,'/predict')
api.add_resource(PredictB64,'/predict_b64')

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 5000)