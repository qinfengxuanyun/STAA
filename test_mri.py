import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
from model.transformer import S_Transformer
from model.ScaleDense import ScaleDense_VAE
from utils.mri_gene_dataset import MRIandGenedataset
from options.test_options import TestOptions
from utils.base_function import  criterion_psnr
from utils.SSIM_loss import SSIM

cuda = torch.cuda.is_available()

opt = TestOptions().parse()
# initial for recurrence
seed = 8
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = False

th = opt.mri_th
alpha_guide = opt.alpha_guide
alpha_mse = opt.alpha_mse
dropout_rate = opt.mask_dropout

MODEL_PATH = f"./generation_models/MRI_S-MRI_{alpha_mse}_{alpha_guide}_{opt.alpha_sparse}_{opt.mri_th}"
model_dict = {0:190,1:190,2:190,3:190,4:190}

# '''!!!!!!!!!!!!!!!!!!manually!!!!!!!!!!!!!!!!!!!!!!!!'''
TEST_BATCH_SIZE = 8
WORKERS = 8
WIDTH = 32
NUM = 1200
NUM2 = 441

criterion_ssim = SSIM().cuda()

for fold in range(5):
    dataset_test = MRIandGenedataset(i=fold,opt="test")
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=True,
                                                    num_workers=WORKERS)
    dataset_size_test = len(data_loader_test)
    print("test size: ", dataset_size_test)
    G = ScaleDense_VAE().cuda()
    G.load_state_dict(torch.load(MODEL_PATH+"/Fold_"+str(fold)+"Epoch"+ str(model_dict[fold])+"_G.pth"))
    G3 = S_Transformer(opt).cuda()   
    G3.load_state_dict(torch.load(MODEL_PATH+"/Fold_"+str(fold)+"Epoch"+ str(model_dict[fold])+"_G3.pth"))

    w_sl = torch.load(MODEL_PATH+"/Fold_"+str(fold)+"Epoch"+ str(model_dict[fold])+"_w_sl.pth").cuda()
    w_sl = w_sl.cuda()
    w_sl.requires_grad = False
    w_sl2 = torch.ones([NUM2,1])
    w_sl2 = w_sl2.cuda()
    w_sl2.requires_grad = False
    for p in G.parameters():
        p.requires_grad = False
    for p in G3.parameters():
        p.requires_grad = False
    
    ###############test######################  
    T_ = 0
    F_ = 0
    G.eval()
    G3.eval() 
    D.eval() 
    prob_all = []
    label_all = []
    predict_class_all = []
    ssim_all = []
    psnr_all = []
    with torch.no_grad():
        for index,test_data in enumerate(data_loader_test):
            fid, label, integer_encoded,age_sex,input = test_data #1500 1536
            
            label = label.cuda()
            B, L = integer_encoded.shape
            
            input = input.cuda() 
            age_sex = age_sex.cuda()
            snp = integer_encoded.cuda()
            ##############################################
            feature, x_list = G(input, out_rec=True)
            x_ = x_list[-1]
            ssim_all.append((1.0 - criterion_ssim(input,x_)).item())
            psnr_all.append(criterion_psnr(input,x_,torch.max(input).item()))
            
            feature = feature.view(B,opt.hidden_size,-1).permute([0,2,1])
            y, gate, mri_feature = G3(w_sl = w_sl,inputs_embeds=feature,width = WIDTH,batch = B)
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

      