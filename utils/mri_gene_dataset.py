import os
import torch
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
from skimage import transform
import csv
import random

def white0(image, threshold=0):
    "standardize voxels with value > threshold"
    image = image.astype(np.float32)
    mask = (image > threshold).astype(int)

    image_h = image * mask
    image_l = image * (1 - mask)

    mean = np.sum(image_h) / np.sum(mask)
    std = np.sqrt(np.sum(np.abs(image_h - mean)**2) / np.sum(mask))

    if std > 0:
        ret = (image_h - mean) / std + image_l
    else:
        ret = image * 0.
    return ret

class MRIandGenedataset(Dataset):
    def __init__(self,i=0,k=5,opt="train",gene_path='ADNI1/SNP_ALL3/',data_path="/home/qinfeng/ADNI/"): #'ADNI1/SNP_ALL3/'
        self.path = data_path
        apoe4_data = pd.read_csv(data_path+"apoe4.csv",usecols=['Subject','Gene']).values.tolist()
        self.apoe4_dict = {}
        for apoe4 in apoe4_data:
            self.apoe4_dict[apoe4[0]] = apoe4[1]
        data = pd.read_csv(data_path+"age_white.csv").values.tolist()
        self.subject = [i[0] for i in data]
        self.age_sex_dict = {}
        self.ad_subject = []
        self.mci_subject = []
        self.cn_subject = []
        for row in data:
            self.age_sex_dict[row[0]] = [row[2],row[3]/100]

        cn_subject = pd.read_csv("CN_list.csv").values.tolist()
        self.cn_subject = [i[0] for i in cn_subject]
        ad_subject = pd.read_csv("AD_list.csv").values.tolist()
        self.ad_subject = [i[0] for i in ad_subject]
        mci_subject = pd.read_csv("MCI_list.csv").values.tolist()
        self.mci_subject = [i[0] for i in mci_subject]

        self.gene_dict = {}
        for sub in self.subject:
            self.gene_dict[sub] = pd.read_csv(data_path+gene_path+sub+".csv",usecols=['value']).values[0:,0].tolist()
           
        self.subject_list = []
        self.label_list = []

        assert k > 1
        fold_size = len(self.cn_subject) // k  
        fold_size2 = len(self.ad_subject) // k 
        fold_size3 = len(self.mci_subject) // k 

        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)  
            idx2 = slice(j * fold_size2, (j + 1) * fold_size2) 
            idx3 = slice(j * fold_size3, (j + 1) * fold_size3)  
            if opt == "train":
                if j is not i:  
                    add_list = self.cn_subject[idx]
                    add_list2 = self.ad_subject[idx2]
                    add_list3 = self.mci_subject[idx3]
                    self.subject_list =  self.subject_list + add_list + add_list2
                    self.label_list =  self.label_list + [0 for index in range(len(add_list))]
                    self.label_list =  self.label_list + [1 for index in range(len(add_list2))]
            elif opt == "all":
                add_list = self.cn_subject[idx]
                add_list2 = self.ad_subject[idx2]
                add_list3 = self.mci_subject[idx3]
                self.subject_list =  self.subject_list + add_list + add_list2
                self.label_list =  self.label_list + [0 for index in range(len(add_list))]
                self.label_list =  self.label_list + [1 for index in range(len(add_list2))]
            else:
                if j == i:  ###第i折作valid
                    add_list = self.cn_subject[idx]
                    add_list2 = self.ad_subject[idx2]
                    add_list3 = self.mci_subject[idx3]
                    self.subject_list =  self.subject_list + add_list + add_list2
                    self.label_list =  self.label_list + [0 for index in range(len(add_list))]
                    self.label_list =  self.label_list + [1 for index in range(len(add_list2))]
        
        # The LabelEncoder encodes a sequence of bases as a sequence of integers.
        self.integer_encoder = LabelEncoder()
        # The OneHotEncoder converts an array of integers to a sparse matrix where
        self.one_hot_encoder = OneHotEncoder(categories='auto')

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        label = np.array(self.label_list[index])
        fid = self.subject_list[index]
        sequence = [self.apoe4_dict[fid]]+self.gene_dict[fid]
        age_sex = np.array(list(map(float,self.age_sex_dict[fid]))).astype(np.float32)
        integer_encoded = self.integer_encoder.fit_transform(sequence)
        integer_encoded = np.array(integer_encoded).astype(np.int64)
        feature = self.get_img(fid)
        return fid, label, integer_encoded, age_sex, feature

    def get_img(self,fid):
        sub_path = self.path+"ADNI1/ALL2/"+fid+".nii.gz"
        img = self.nii_loader(sub_path)
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype= np.float32)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        return img

    def nii_loader(self, path):
        img = nib.load(str(path))
        data = img.get_fdata()
        return data
