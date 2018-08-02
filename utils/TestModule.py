from __future__ import print_function
import torch
from PIL import Image
from torch.autograd import Variable
import numpy as np
#import argparse
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

import pdb
import time
import os

class TestModule:
    model = 0;
    data_transforms = 0;
    
    def __init__(self, model, data_transforms, dataPath , labelPath, imIdPath, fiftyPatch):
        self.model = model;
        self.data_transforms = data_transforms;
        self.dataPath = dataPath;
        self.labelPath = labelPath;
        self.imIdPath = imIdPath;
        self.startTime = time.time();
        self.fiftyPatch = fiftyPatch;
            
    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        #pdb.set_trace();
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img_array = np.asarray(img);
                if len(img_array.shape) < 3 or img_array.shape[2] <= 3:
                    return img.convert('RGB')
                else:
                    return img.copy();    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))      
       
    #predicts the argmax index
    def PredictAccuracyFromPath(self,imPath):
        #piexif.remove(imPath);
        img = Image.open(imPath);
        img_tensor = self.data_transforms(img);
        outputs = self.predict(img_tensor);
       # _, preds = torch.max(outputs.data, 1);
        preds = np.argmax(outputs);
        return preds;
        
    #Finds wTx for one random crop
    def PredictwTx(self,imPath):
        img = self.pil_loader(imPath);
        img_tensor = self.data_transforms(img);
        outputs = self.predict(img_tensor);        
        return outputs;
    

    #Predicts the transformed image tensor        
    def predict(self,img_tensor):
        img_variable = Variable(img_tensor.cuda());
        outputs = self.model(img_variable.unsqueeze(0));
        return outputs.data.cpu().numpy()[0];
                    
            
    def Cuda_predict(self, img_tensors):
        #pdb.set_trace();
        img_variable = Variable(img_tensors.cuda());
        outputs = self.model(img_variable);
        return outputs.data.cpu().numpy();
        
        
    def Cuda_PredictAveragewTx(self,imPath,batchSize):
        img = self.pil_loader(imPath);
        imageCrops = [];
        scores = [];
        for i in range(50):
            img_tensor = self.data_transforms(img);
            imageCrops.append(img_tensor);
        #pdb.set_trace();
        batchIndex = 0;
        while batchIndex < len(imageCrops):
            #scores = np.vstack((scores,self.Cuda_predict(torch.stack(imageCrops[batchIndex:batchIndex+batchSize]))));
            scores.append(self.Cuda_predict(torch.stack(imageCrops[batchIndex:batchIndex+batchSize])));            
            batchIndex = batchIndex+batchSize;
        #pdb.set_trace();
        #scores = np.delete(scores, 0, 0);
        #pdb.set_trace();
        scores = np.vstack(scores);
        #scores = self.Cuda_predict(torch.stack(imageCrops));
        return np.mean(scores, axis = 0)
            
        
        
    def MAPTracker(self):
        print ("\nTesting MAP... ");
        testLabels = np.loadtxt(self.labelPath, dtype = int);
        testIds = np.loadtxt(self.imIdPath,dtype = int);
        result = []; # array to store the wTx values
        corruptInt = []; # list of corrupted files
        #pdb.set_trace();
        for c,id in enumerate(testIds):
            if os.path.exists(self.dataPath + str(id)+'.png'):
                imPath = self.dataPath + str(id)+'.png';
            if os.path.exists(self.dataPath + str(id)+'.jpg'):
                imPath = self.dataPath + str(id)+'.jpg';
                
            #prediction = self.Cuda_PredictAveragewTx(imPath, 25);
            #prediction = self.PredictAveragewTx(imPath);
            try:
                if self.fiftyPatch:
                    prediction = self.Cuda_PredictAveragewTx(imPath, 24);
                else:
                    prediction = self.PredictwTx(imPath);
            except Exception as e:
                print("Could not process %s \n Error : %s"%(imPath, str(e)));
                corruptInt.append(c);
                continue;
            result.append(prediction);
            if (c%500 == 0):
                print("%d / %d images processed."%(c,len(testIds)))
        #remove corrupt labels
        testLabels = testLabels;
        testLabels = np.delete(testLabels,corruptInt,0);
        #pdb.set_trace();
        #result = result[::50];
        
        
        print ("Prediction Complete... \n\nNumber of samples used for testing = %d \nNumber of GT values = %d\n"
        %(len(result),testLabels.shape[0]));
        #pdb.set_trace();
        PerClassP = average_precision_score(testLabels,np.array(result), average=None);
        AP = average_precision_score(testLabels,np.array(result), average='samples');
        mAP_macro = average_precision_score(testLabels,np.array(result), average='macro');
        mAP_weighted = average_precision_score(testLabels,np.array(result), average='weighted');
        mAP_micro = average_precision_score(testLabels,np.array(result), average='micro');
        CM = self.CMeter(testLabels, result);
        #pdb.set_trace();
        print ("Run Time : %f seconds"%(time.time() - self.startTime));
        return CM, AP, mAP_macro, mAP_weighted, mAP_micro, PerClassP;


    def CMeter(self, testLabels, result, multilabel = False ):
        if testLabels.shape[1] == 14:
            multilabel = True;
        if not multilabel:
            #pdb.set_trace();    
            return confusion_matrix(np.argmax(testLabels, axis = 1), np.argmax(result, axis = 1));
        else :
            conf_mat = np.zeros([testLabels.shape[1],testLabels.shape[1]]);
            for gt, pl in zip(testLabels, result):
                #pdb.set_trace();
                gtLabels = np.where(gt == 1)[0];
                predLabels = np.argsort(pl)[::-1][0:gtLabels.shape[0]];
                matches = np.intersect1d(gtLabels, predLabels);
                for m in matches:
                    conf_mat[m][m]+=1;
                for g in gtLabels:
                    if g not in matches:
                        for p in predLabels:
                            if p not in matches:
                                conf_mat[g][p]+=1;
        return conf_mat;  
        
    def PredictSortedLabels(self,imPath):
        scores = self.PredictwTx(imPath);
        sortedScores = np.argsort(-scores);
        labels = ['Complementary_Colors', 'Duotones', 'HDR', 'Image_Grain', \
        'Light_On_White', 'Long_Exposure', 'Macro', 'Motion_Blur', 'Negative_Image',\
        'Rule_of_Thirds', 'Shallow_DOF', 'Silhouettes', 'Soft_Focus', 'Vanishing_Point']
        predictedLabels = [];        
        for i in sortedScores:
            predictedLabels.append(labels[i]);
        return zip(predictedLabels,self.sigmoid(scores[sortedScores]).tolist());