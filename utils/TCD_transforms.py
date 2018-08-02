from __future__ import division
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import collections

import cv2
from torchvision import transforms
            

class Sal_Resize(object):
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size        
        
    def __call__(self, img):
        #pdb.set_trace()
        img_array = np.array(img)
        new_image = cv2.resize(img_array, tuple(self.size));
        return Image.fromarray(new_image)

class Sal_CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
            
        self.CenterCrop = transforms.CenterCrop(self.size)
        self.Resize = transforms.Resize(self.size)
        
    def __call__(self, img):        
        R, G, B, A = img.split()
        return Image.merge('RGBA', self.CenterCrop(Image.merge('RGB',(R,G,B))).split() + (self.Resize(A),))
        
class Sal_RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
            
        self.RandomCrop = transforms.RandomCrop(self.size, pad_if_needed = True)
        self.Resize = transforms.Resize(self.size)
        
    def __call__(self, img):        
        R, G, B, A = img.split()
        return Image.merge('RGBA', self.RandomCrop(Image.merge('RGB',(R,G,B))).split() + (self.Resize(A),))        
        
        
class Sal_RandomResizedCrop(object):
    def __init__(self, size):
        self.size = size            
        self.RandomResizedCrop = transforms.RandomResizedCrop(self.size)
        self.Resize = transforms.Resize([self.size, self.size])
        
    def __call__(self, img):        
        R, G, B, A = img.split()
        return Image.merge('RGBA', self.RandomResizedCrop(Image.merge('RGB',(R,G,B))).split() + (self.Resize(A),))        
        
    
class Rapid_Crop_Scale(object):    
    
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
            
    def __call__(self, img):
        #pdb.set_trace();
        random_crop = transforms.RandomCrop(self.size);
        scale = transforms.Resize(self.size);
        t_rc = random_crop(img);
        t_scale = scale(img);
        final_image = np.dstack((t_rc,t_scale));
        #pdb.set_trace();
        return final_image;