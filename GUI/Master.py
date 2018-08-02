from flask import Flask, request, render_template
import urllib
import torch
import random
import string
import sys
import os

#adds the utl folder to the path
sys.path.append('../utils');

#imports files from utils folder
import TestModule
import Augmentation
#import time

model = torch.load(open(sys.argv[1],'r'))
app = Flask(__name__)

 
@app.route("/")
def hello():
    return render_template('index.html')
 
@app.route("/runTest", methods=['POST'])
def runTest():
    global model    
    aug = Augmentation.Augmentation(sys.argv[2]);
    data_transforms = aug.applyTransforms();
    imageLink = request.form['link'];
    timestr = ''.join(random.choice(string.lowercase) for x in range(8));
    dst = "static/augdata/"+timestr+'.jpg';
    float_formatter = lambda x: "\t %.3f" % x;
    try:
        urllib.urlretrieve(imageLink, dst);
    except:
        print ('Cannot find %s'%(dst))
        return render_template('index.html')
    #time.sleep(3);
    #model = torch.load("/home/koustav/Desktop/IPA/intelligent_photograph_assesment/CNN/models/net_DenseNet161_crop_True_lr_0.001.model");
    t =   TestModule.TestModule(model,data_transforms['val'],'','','','');
    namevaluepairs = t.PredictSortedLabels(dst);
    for count, (style, prob) in enumerate(namevaluepairs):
            namevaluepairs[count] = (style,float_formatter(prob));
    #styleandvalues = "\n".join('{}:   {}'.format(val,key) for key, val in namevaluepairs.items())
    return render_template('index.html', testImageName = dst, styleLabels = namevaluepairs );

    #return render_template('index.html', testImageName = dst);

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response



if __name__ == "__main__":
#    app.run(host= 'localhost',debug = True)
     app.run(host= '0.0.0.0', port =3134, debug = True)
    
