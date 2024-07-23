import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from torch.nn import functional as F
import random
from sklearn.metrics import roc_auc_score, confusion_matrix

from model.transformer import S_Transformer
from model.ScaleDense import ScaleDense_VAE
from options.test_options import TestOptions
from utils.mri_gene_dataset import MRIandGenedataset

cuda = torch.cuda.is_available()

opt = TestOptions().parse()

# initial setup # '''!!!!!!!!!!!!!!!!!!manually!!!!!!!!!!!!!!!!!!!!!!!!'''
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
lr = 1e-4
EPOCH = 60
WORKERS = 8

WIDTH = 32
NUM = 1200
NUM2 = 441

MODEL_PATH_STAGE1 = f"./generation_models/MRI_S-MRI_{opt.alpha_mse}_{opt.alpha_guide}_{opt.alpha_sparse}_{opt.mri_th}"
model_dict = {0:190,1:190,2:190,3:190,4:190}
MODEL_PATH_STAGE2 = f'./generation_models/MRI_Gene2MRI_{opt.alpha_mse2}_{opt.alpha_guide2}_{opt.alpha_sparse2}_{opt.mri_th}_{opt.snp_th}'
model_dict = {0:30,1:30,2:30,3:30,4:30}

for fold in range(5):
    # if fold==0:
    #     continue
    seed = 8
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    dataset_test = MRIandGenedataset(i=fold,opt="test")
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=True,
                                                    num_workers=WORKERS)
    dataset_size_test = len(data_loader_test)
    
    G = ScaleDense_VAE().cuda()
    G.load_state_dict(torch.load(MODEL_PATH_STAGE1+"/Fold_"+str(fold)+"Epoch"+ str(model_dict[fold])+"_G.pth"))
    G3 = S_Transformer(opt).cuda()
    G3.load_state_dict(torch.load(MODEL_PATH_STAGE1+"/Fold_"+str(fold)+"Epoch"+ str(model_dict[fold])+"_G3.pth"))
    G2 = S_Transformer(opt,out_mask=True).cuda()
    G2.load_state_dict(torch.load(MODEL_PATH_STAGE2+"/Fold_"+str(fold)+"Epoch"+ str(model_dict[fold])+"_G2.pth"))

    w_sl = torch.load(MODEL_PATH_STAGE1+"/Fold_"+str(fold)+"Epoch"+ str(model_dict[fold])+"_w_sl.pth").cuda()
    w_sl.requires_grad = False
    w_sl2 = torch.load(MODEL_PATH_STAGE2+"/Fold_"+str(fold)+"Epoch"+ str(model_dict[fold])+"_w_sl.pth").cuda()
    w_sl2.requires_grad = False
    
    T_ = 0
    F_ = 0
    G.eval()
    G2.eval() 
    G3.eval()
    prob_all = []
    label_all = [] 
    predict_class_all = [] 
    with torch.no_grad():
        for index,test_data in enumerate(data_loader_test):
            fid, label, integer_encoded,age_sex,input = test_data #1500 1536 
            label = label.cuda()
            B, L = integer_encoded.shape
            
            input = input.cuda()
            age_sex = age_sex.cuda()
            snp = integer_encoded.cuda()
            ##############################################
            _, gate, g_feature, mask = G2(input_ids=snp,w_sl = w_sl2,width = WIDTH,batch = B, age_sex=age_sex)
            g_feature = g_feature.view(B,512,-1).permute([0,2,1]) 
            
            y, gate2, mri_feature = G3(w_sl = w_sl,inputs_embeds=g_feature,width = WIDTH,batch = B)
            
            out_c = F.softmax(y, dim=1)
            _, predicted = torch.max(out_c.data, 1)
            PREDICTED_ = predicted.data.cpu().numpy()
            REAL_ = label.data.cpu().numpy()

            for k in range(PREDICTED_.shape[0]):
                if PREDICTED_[k] == REAL_[k]:
                    T_ += 1
                else:
                    F_ += 1
            prob_all.extend(out_c[:,1].cpu().numpy())
            label_all.extend(label.cpu().numpy())
            predict_class_all.extend(PREDICTED_)   
            
    test_acc = T_  / (T_ + F_)
    test_auc = roc_auc_score(label_all,prob_all)
    test_cm = confusion_matrix(label_all,predict_class_all)
    test_sen = test_cm[0,0]/(test_cm[0,0]+test_cm[0,1])
    test_spe = test_cm[1,1]/(test_cm[1,0]+test_cm[1,1])
    print("test_acc:", test_acc)
    print("test_auc:", test_auc)
    print("test_sen:", test_sen)
    print("test_spe:", test_spe)
