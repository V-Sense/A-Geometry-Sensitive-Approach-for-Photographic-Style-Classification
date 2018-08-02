from __future__ import print_function

#adds the utl folder to the path
from torchvision import transforms
import TCD_transforms
class Augmentation:
    
    def __init__(self,strategy):
        print ("Data Augmentation Initialized with strategy %s"%(strategy));
        self.strategy = strategy;
        #pdb.set_trace()
        
    
    def cropAtLocation (self, x,y,z): return lambda p : p.crop((x-z/2,y-z/2,x+z/2,y+z/2))
    
    def applyTransforms(self, crop_x = 0, crop_y = 0, crop_size = 224):
        #print (model)
        #pdb.set_trace()
        if self.strategy == "RC": # Random Crops of 224
            data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        elif self.strategy == "RSC": 
            data_transforms = {
            'train': transforms.Compose([
                transforms.RandomSizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.RandomSizedCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        elif self.strategy == "CC": 
            data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        elif self.strategy == "ICC": 
             data_transforms = {
            'train': transforms.Compose([
                transforms.Scale(crop_size),
                transforms.CenterCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Scale(crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        elif self.strategy == "S": 
            data_transforms = {
            'train': transforms.Compose([
                transforms.Scale([crop_size,crop_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Scale([crop_size,crop_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        elif self.strategy == "SAL_S": 
             data_transforms = {
            'train': transforms.Compose([                
                TCD_transforms.Sal_Resize([crop_size,crop_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.063], [0.229, 0.224, 0.225, 0.118 ])
            ]),
            'val': transforms.Compose([                
                TCD_transforms.Sal_Resize([crop_size,crop_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.063], [0.229, 0.224, 0.225, 0.118 ])
            ]),
     }
        
        elif self.strategy == "SAL_CC": 
             data_transforms = {
            'train': transforms.Compose([                
                TCD_transforms.Sal_CenterCrop([crop_size,crop_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.063], [0.229, 0.224, 0.225, 0.118 ])
            ]),
            'val': transforms.Compose([                
                TCD_transforms.Sal_CenterCrop([crop_size,crop_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.063], [0.229, 0.224, 0.225, 0.118 ])
            ]),
     }
        
        elif self.strategy == "SAL_RC":
            data_transforms = {
            'train': transforms.Compose([                
                TCD_transforms.Sal_RandomCrop([crop_size,crop_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.063], [0.229, 0.224, 0.225, 0.118 ])
            ]),
            'val': transforms.Compose([                
                TCD_transforms.Sal_RandomCrop([crop_size,crop_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.063], [0.229, 0.224, 0.225, 0.118 ])
            ]),
        }

        elif self.strategy == "SAL_RSC":
            data_transforms = {
            'train': transforms.Compose([                
                TCD_transforms.Sal_RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.063], [0.229, 0.224, 0.225, 0.118 ])
            ]),
            'val': transforms.Compose([                
                TCD_transforms.Sal_RandomResizedCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.063], [0.229, 0.224, 0.225, 0.118 ])
            ]),
        }
        elif self.strategy == "SAL_ICC":
            data_transforms = {
            'train': transforms.Compose([                
                TCD_transforms.Sal_Resize([crop_size,crop_size]),
                TCD_transforms.Sal_CenterCrop([crop_size,crop_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.063], [0.229, 0.224, 0.225, 0.118 ])
            ]),
            'val': transforms.Compose([                
                TCD_transforms.Sal_Resize([crop_size,crop_size]),
                TCD_transforms.Sal_CenterCrop([crop_size,crop_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.063], [0.229, 0.224, 0.225, 0.118 ])
            ]),
        }

        elif self.strategy == "RAPID_CROP_SCALE": # Same as RAPID 2 col
            data_transforms = {
            'train': transforms.Compose([
                TCD_transforms.Rapid_Crop_Scale(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.063], [0.229, 0.224, 0.225, 0.118 ])
            ]),
            'val': transforms.Compose([
                TCD_transforms.Rapid_Crop_Scale(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.063], [0.229, 0.224, 0.225, 0.118 ])
            ]),
        }       
     

        
        
        
        else :
            print ("Please specify correct augmentation strategy RC, RSC, CC, SC, S");
            exit();
            
        return data_transforms;