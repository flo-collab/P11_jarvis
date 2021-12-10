#from __future__ import division, print_function
import sys
import os
import glob
import re

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from torchimport import *
from torchimport import MedNet

list_stuff_path = 'list_stuff.sav' 
list_stuff = pickle.load(open(list_stuff_path, 'rb'))
dataDir,classNames,numClass,imageFiles,nnumEach,imageFilesList,imageClass,numTotal,imageWidth, imageHeight = list_stuff[0],list_stuff[1],list_stuff[2],list_stuff[3],list_stuff[4],list_stuff[5],list_stuff[6],list_stuff[7],list_stuff[8],list_stuff[9]

model = MedNet(imageWidth,imageHeight,numClass).to(dev)
model = torch.load('saved_model_for_zobi', map_location=torch.device('cpu'))

app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def make_predict(img,model):
    tensor_img = scaleImage(Image.open(img))
    tensor_img = tensor_img[None,:]

    with torch.no_grad():
        stuff = model(tensor_img.to(dev))
    result = stuff.max(1,keepdim=True)[1][0][0].tolist()
    predict = classNames[result]

    return predict

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        result_func = make_predict(f,model)

    return result_func

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)

