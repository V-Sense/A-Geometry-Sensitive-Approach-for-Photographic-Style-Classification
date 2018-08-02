from __future__ import print_function, division
import pdb
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import dataloader
import time
import os
import sys
import csv
import datetime
from termcolor import colored
#custom modules
sys.path.append('utils');

import Augmentation as ag
import GhosalkImageFolder as gk
#pdb.set_trace();
import TestModule
import Models
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

##################  For handling corrupted images #####################
def my_collate(batch):
    batch = filter (lambda x:x is not None, batch)
    return dataloader.default_collate(batch)
    
class ErrorHandlingImageFolder(gk.MyImageFolder):
    __init__ = gk.MyImageFolder.__init__
    def __getitem__(self, index):
        try: 
            return super(ErrorHandlingImageFolder, self).__getitem__(index)
        except Exception as e:
            print(e)
#######################################################################        
########################## Command Line Parsers #######################
parser = argparse.ArgumentParser(description='AVA Style Classification');

#Training arguments
parser.add_argument('--tag', type=str, default='__default__',
                    help='name of the experiment for saving models etc')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPUs for training')                    
parser.add_argument('--datapath', type=str, default='',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='',
                    help='Name of Convolutional Net (DenseNet161, DenseNet121, \
                    ResNet152, ResNet101, ResNet50, ResNet30, ResNet18, VGG19, \
                    VGG16, Resnet_Saliency_Direct, Densenet_Saliency_Direct)')
parser.add_argument('--pooling', action='store_true',
                    help='perform max pooling . if false performs strided convolutions')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default = 16, metavar='N',
                    help='batch size')
parser.add_argument('--save', type=str,  default='',
                    help='path to save the final model')
parser.add_argument('--aug_train', type=str, default = 'ICC',
                    help='training data augmentation strategy')
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model')
                    
#Testing arguments
parser.add_argument('--withTesting', action='store_true',
                    help='perform testing')
parser.add_argument('--aug_test', type=str, default = '',
                    help='testing data augmentation strategy')                    
parser.add_argument('--fiftyPatch', action='store_true',
                    help='use 50 patch testing')
parser.add_argument('--csv', type=str, default='Results.csv',
                    help='csv to save data')
parser.add_argument('--testDataPath', type=str, default='',help='Path to testdata')
parser.add_argument('--testLabels', type=str, default='',help='Path to testLabels')
parser.add_argument('--testIds', type=str, default='',help='Path to testIds')
#pdb.set_trace();
args = parser.parse_args();
arg_str = '\n'.join(sorted([str(i) + ' : ' + str(j) for (i,j) in vars(args).items()]))
print (colored(arg_str,'white'))
##########################################################################
#################### naming the model ####################################
model_path = args.save+'net_'+'_dataset_' + args.tag + '_model_'+ args.model +\
'_pretrained_'+str(args.pretrained)+ '_aug_'+str(args.aug_train)+\
'_'+str(args.aug_test)+'_lr_'+str(args.lr)+'.model';
###########################################################################

#Data augmentation initialized
DAug = ag.Augmentation(args.aug_train);
data_transforms = DAug.applyTransforms();


##################### Loading Data ######################
data_dir = args.datapath;
dsets = {x: ErrorHandlingImageFolder(os.path.join(data_dir, x), data_transforms[x])
             for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,
                                               shuffle=True, num_workers=8, collate_fn=my_collate)
                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes
##########################################################
use_gpu = True;


############ Train function ################################
def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()

    best_acc = 0.0
    global model_path;    
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            #pdb.set_trace();
            # Iterate over data.
            for count, data in enumerate(dset_loaders[phase]):
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                #pdb.set_trace();                
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                #pdb.set_trace();
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics                
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                
                #if count%10 == 0:
                #    print('Batch %d / %d || Running Loss = %0.6f || Running Accuracy = %0.6f'%(count+1, len(dset_loaders[phase]),running_loss/(args.batch_size*(count+1)),(running_corrects*100)/(args.batch_size*(count+1))))
                

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = (running_corrects*100) / dset_sizes[phase]
            

            print('Epoch %d || %s Loss: %.4f || Acc: %.4f'%(epoch,
                phase, epoch_loss, epoch_acc),end = ' || ')
            if phase == 'val':
                print ('\n', end='');
                lr_scheduler.step(epoch_loss);
                
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                
                with open(model_path, 'wb') as f:
                    torch.save(model, f);

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return 1

#Finetuning
#pdb.set_trace();

################## List of models ########################################
if args.model == "DenseNet161":
    model_ft = models.densenet161(args.pretrained);
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features,len(dset_classes));

elif args.model == "DenseNet121":
    model_ft = models.densenet121(args.pretrained);
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features,len(dset_classes));


elif args.model == "ResNet152" :
    model_ft = models.resnet152(args.pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(dset_classes));


elif args.model == "ResNet18" :
    model_ft = models.resnet18(args.pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(dset_classes));

elif args.model == "ResNet34" :
    model_ft = models.resnet34(args.pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(dset_classes));

elif args.model == "ResNet50" :
    model_ft = models.resnet50(args.pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(dset_classes));

elif args.model == "ResNet101" :
    model_ft = models.resnet101(args.pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(dset_classes));

elif args.model == "VGG19":
    model_ft = models.vgg19(args.pretrained);
    model_ft.classifier._modules['6'] = nn.Linear(4096, len(dset_classes));
    
elif args.model == "VGG16":
    model_ft = models.vgg16(args.pretrained);
    model_ft.classifier._modules['6'] = nn.Linear(4096, len(dset_classes));
   
elif args.model == "ResNet_Saliency_Direct" :
    model_ft = Models.resnet_saliency_direct(args.pretrained, args.pooling);
    num_ftrs = model_ft.fc.in_features
    model_ft.net.fc = nn.Linear(num_ftrs, len(dset_classes));
    if args.multi_gpu:
        model_ft = nn.DataParallel(model_ft);  
    
elif args.model == "DenseNet_Saliency_Direct" :
    model_ft = Models.densenet_saliency_direct(args.pretrained, args.pooling);
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(dset_classes));
    if args.multi_gpu:
        model_ft = nn.DataParallel(model_ft);     
else :
    print (colored ("ERROR : Model %s not found. Use the correct definition for --model below"%(args.model), 'red'))    
    parser.print_help()
    exit();    
#################################################################################
if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()
#pdb.set_trace()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9)
#pdb.set_trace();
scheduler_ft = ReduceLROnPlateau(optimizer_ft, 'min', factor = 0.5,patience = 3, verbose = True)

if os.path.exists(model_path):
    print('Loading CNN from %s'%(model_path))
    model_ft = torch.load(model_path);

print ('Training Started...')
train_model(model_ft, criterion, optimizer_ft, scheduler_ft,
                      num_epochs=args.epochs);

#Testing
if args.withTesting:
    DAug = ag.Augmentation(args.aug_test);
    data_transforms = DAug.applyTransforms();
    t =   TestModule.TestModule(model_ft,data_transforms['val'], args.testDataPath, args.testLabels, args.testIds, args.fiftyPatch);
    
    CM, AP,mAP_macro,mAP_weighted, mAP_micro, PerClassP = t.MAPTracker();
    printed_results = '\n'.join(["AP : "+str(AP), "MAP_MACRO : "+str(mAP_macro), "MAP_MICRO : "+str(mAP_micro), "MAP_WEIGHTED : "+str(mAP_weighted)])
    PCP = '\n'.join([cname + ':' + str(pcp) for cname, pcp in zip(dset_classes, PerClassP.tolist() )])    
    #print ("\nAP = %f \nmAP_macro = %f \nmAP_weighted = %f \nmAP_micro = %f \n"%(AP,mAP_macro,mAP_weighted,mAP_micro));

    print ("\nPrecision\n######################")
    print(colored(printed_results, 'white'))
        
    print ("\nPer Class Precision\n######################")    
    print(colored(PCP, 'white'))    
#    print ("Per Class : ",PerClassP);
    print ("\nConfusion :", CM);
    results = [datetime.datetime.now(), args.tag, args.model, args.epochs, args.batch_size, args.aug_train, args.aug_test, \
    str(args.pretrained), args.lr , AP, mAP_macro,mAP_weighted, mAP_micro ] + PerClassP.tolist()
    results = [str(i) for i in results];
    parameters = ['Time','TAG','Model', 'Epochs', 'Batch_Size', 'Aug-Train','Aug-Test','Pre-trained', 'LR', 'AP',\
    'MAP_MACRO', 'MAP_WEIGHTED', 'MAP_MICRO']+dset_classes
    #pdb.set_trace();
    
    if not os.path.exists(args.csv):
        with open(args.csv, 'a', os.O_NONBLOCK) as f:
            writer = csv.writer(f)
            writer.writerow(parameters);
        
    with open(args.csv, 'a', os.O_NONBLOCK) as f:
        writer = csv.writer(f)
        writer.writerow(results);
