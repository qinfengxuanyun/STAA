import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import time
import math
import random
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, confusion_matrix

from model.transformer import S_Transformer
from model.ScaleDense import ScaleDense_VAE,ScaleDense_Dis
from options.train_options import TrainOptions
from utils.mri_gene_dataset import MRIandGenedataset
from utils.base_function import GANLoss
from utils.SSIM_loss import SSIM
from utils.base_function import _freeze,_unfreeze

cuda = torch.cuda.is_available()

opt = TrainOptions().parse()
# initial for recurrence
criterion_bce = nn.BCELoss().cuda()
criterion_l1 = nn.L1Loss().cuda()
criterion_mse = nn.MSELoss().cuda()
criterion_ssim = SSIM().cuda()
criterion = nn.CrossEntropyLoss().cuda()
gan_loss = GANLoss('lsgan')

# initial setup # '''!!!!!!!!!!!!!!!!!!manually!!!!!!!!!!!!!!!!!!!!!!!!'''
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
lr = 1e-4
EPOCH = 60
WORKERS = 8

WIDTH = 32
NUM = 1200
NUM2 = 441

acc_th = 0.75
clamp_th = opt.clamp_th
alpha_guide = opt.alpha_guide2
alpha_mse = opt.alpha_mse2

MODEL_PATH_STAGE1 = f"./generation_models/MRI_S-MRI_{opt.alpha_mse}_{opt.alpha_guide}_{opt.alpha_sparse}_{opt.mri_th}"
model_dict = {0:190,1:190,2:190,3:190,4:190}
MODEL_PATH = f'./generation_models/MRI_Gene2MRI_{alpha_mse}_{alpha_guide}_{opt.alpha_sparse2}_{opt.mri_th}_{opt.snp_th}'
LOG_PATH = f'./logs/log_gene2mri_{alpha_mse}_{alpha_guide}_{opt.alpha_sparse2}_{opt.mri_th}_{opt.snp_th}'
os.system("mkdir -p {}".format(MODEL_PATH))
os.system("mkdir -p {}".format(LOG_PATH))

writer = SummaryWriter(logdir=LOG_PATH, comment='Gene2MRI')

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
    
    dataset_train = MRIandGenedataset(i=fold,opt="train")
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                                    num_workers=WORKERS)
    dataset_test = MRIandGenedataset(i=fold,opt="test")
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=True,
                                                    num_workers=WORKERS)
    dataset_size = len(data_loader_train)
    dataset_size_test = len(data_loader_test)
    
    G = ScaleDense_VAE().cuda()
    G.load_state_dict(torch.load(MODEL_PATH_STAGE1+"/Fold_"+str(fold)+"Epoch"+ str(model_dict[fold])+"_G.pth"))
    G3 = S_Transformer(opt).cuda()
    G3.load_state_dict(torch.load(MODEL_PATH_STAGE1+"/Fold_"+str(fold)+"Epoch"+ str(model_dict[fold])+"_G3.pth"))
    D = ScaleDense_Dis().cuda()
   
    G2 = S_Transformer(opt,out_mask=True).cuda()

    w_sl =torch.load(MODEL_PATH_STAGE1+"/Fold_"+str(fold)+"Epoch"+ str(model_dict[fold])+"_w_sl.pth").cuda()
    w_sl.requires_grad = False
    w_sl2 = torch.ones([NUM2,1]).cuda()
    w_sl2.requires_grad = False
    
    attention_mask = torch.from_numpy(np.load(MODEL_PATH_STAGE1+f"/feature_mask{fold}_{opt.mri_th}.npy").flatten()).unsqueeze(0).unsqueeze(2).cuda().detach()
    mri_mask = torch.from_numpy(np.load(MODEL_PATH_STAGE1+f"/mri_mask{fold}_{opt.mri_th}.npy")).unsqueeze(0).unsqueeze(1).cuda().detach()
    attention_mask = torch.clamp(attention_mask,min=clamp_th)
    mri_mask = torch.clamp(mri_mask,min=clamp_th)

    print("attention_mask_max: ",torch.max(attention_mask).item())
    print("mri_mask_max: ",torch.max(mri_mask).item())
    for p in G.parameters():
        p.requires_grad = False
    for p in G2.parameters():
        p.requires_grad = True
    for p in G3.parameters():
        p.requires_grad = False
    for p in D.parameters():
        p.requires_grad = True

    g_optimizer = optim.AdamW([{'params': G2.parameters()}], lr=lr)
    w_optimizer = optim.AdamW([{'params': w_sl2, 'weight_decay': 0}], lr=5e-3) 
    d_optimizer = optim.AdamW([{'params': D.parameters()}], lr=lr*0.1) 
    sparse_flag = 0
    lr_changed = 0
    for iteration in range(EPOCH):
        ###############train######################
        print("learning_rate:",g_optimizer.param_groups[0]['lr'])
        print("learning_rate:",w_optimizer.param_groups[0]['lr'])
        print(iteration + 1)
        start_time = time.time() 
        G.eval()   
        G2.train() 
        G3.train()
        D.train()
        total_loss = 0
        total_loss_fmse = 0 
        total_loss_mse = 0 
        total_loss_norm1 = 0 
        total_loss_guide = 0
        total_loss_ssim = 0
        total_loss_mae = 0 
        total_loss_d_ad = 0
        total_loss_g_ad = 0
        T_ = 0
        F_ = 0
        for index,train_data in enumerate(data_loader_train):
            fid, label, integer_encoded,age_sex,input = train_data #1500 1536
            
            label = label.cuda()
            B, L = integer_encoded.shape
            
            input = input.cuda()
            age_sex = age_sex.cuda()
            snp = integer_encoded.cuda()
            ##############################################
            feature = G(input, out_rec=False)
            feature = feature.view(B,opt.hidden_size,-1).permute([0,2,1])
            _, gate, g_feature, mask = G2(input_ids=snp,w_sl = w_sl2,width = WIDTH,batch = B,age_sex=age_sex)
            
            x_1 =  G.up1(g_feature)
            x_2 =  G.up2(x_1)
            x_3 =  G.up3(x_2)
            x_list = []
            x_list.append(G.out1(x_1))
            x_list.append(G.out2(x_2))
            x_list.append(G.out3(x_3))  
            x_ = x_list[-1]
            
            g_feature = g_feature.view(B,512,-1).permute([0,2,1])
            y, gate2, mri_feature = G3(w_sl = w_sl, inputs_embeds=g_feature,width = WIDTH,batch = B)
            
            #discrimator update
            _unfreeze(D)
            D_real = D(input.detach())
            D_fake = D(x_.detach())
            loss_d_ad = (gan_loss(D_real, True, True) + gan_loss(D_fake, False, True))*0.5#*lambda_g
            d_optimizer.zero_grad()
            loss_d_ad.backward()
            d_optimizer.step()

            _freeze(D) 
            D_fake = D(x_)
            loss_g_ad = gan_loss(D_fake, True, False) 

            #W update
            loss_guide = criterion(y, label)
            loss_norm1 = torch.norm(gate,p=1) 
            loss_fmse = torch.mean(torch.sum(torch.pow(feature.detach()-g_feature,2) * attention_mask,dim=1,keepdim=True)/torch.sum(attention_mask))
            loss_mse =  torch.mean(torch.sum(torch.pow(input.detach()-x_,2) * mri_mask,dim=[2,3,4],keepdim=True)/torch.sum(mri_mask))
            # loss_fmse = criterion_mse(feature.detach(),g_feature)
            # loss_mse = criterion_mse(input.detach(),x_)
            if sparse_flag:
                loss_g =  (10**(opt.alpha_sparse2)) * F.relu(loss_norm1-opt.snp_th) + loss_g_ad + alpha_guide*loss_guide + alpha_mse*(loss_mse+loss_fmse)
            else:
                loss_g =  loss_g_ad + alpha_guide*loss_guide + alpha_mse*(loss_mse+loss_fmse)
            if not lr_changed and round(loss_norm1.item()) <= opt.snp_th:
                adjust_learning_rate(w_optimizer, 1e-3)
                lr_changed = 1
            g_optimizer.zero_grad()       
            w_optimizer.zero_grad()
            loss_g.backward()
            w_optimizer.step()
            g_optimizer.step()
           
            total_loss_mse += loss_mse.item()
            total_loss_fmse += loss_fmse.item()
            total_loss_norm1 += loss_norm1.item()
            total_loss_guide += loss_guide.item()
            total_loss_d_ad += loss_d_ad.item()
            total_loss_g_ad += loss_g_ad.item()

        G.eval()
        G2.eval() 
        G3.eval()
        D.eval()  
        prob_all = []
        label_all = []  
        with torch.no_grad():
            for index,train_data in enumerate(data_loader_train):
                fid, label, integer_encoded,age_sex,input = train_data #1500 1536 
                label = label.cuda()
                B, L = integer_encoded.shape
                
                input = input.cuda()
                age_sex = age_sex.cuda()
                snp = integer_encoded.cuda()
                ##############################################
                _, gate, g_feature, mask = G2(input_ids=snp,w_sl = w_sl2,width = WIDTH,batch = B,age_sex=age_sex)
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
                
        train_acc = T_  / (T_ + F_)
        train_auc = roc_auc_score(label_all,prob_all)

        print("loss_guide:",total_loss_guide/dataset_size)
        print("loss_norm1:",total_loss_norm1/dataset_size)
        print("loss_mse:",total_loss_mse/dataset_size)
        print("loss_fmse:",total_loss_fmse/dataset_size)
        print("loss_g_ad:",total_loss_g_ad/dataset_size)
        print("loss_d:",total_loss_d_ad/dataset_size)
        print("loss_ssim:",total_loss_ssim/dataset_size)
        print("loss_mae:",total_loss_mae/dataset_size)
        print("train_acc:",train_acc)
        print("train_auc:",train_auc)

        writer.add_scalars('loss_guide', {'loss_guide'+str(fold):total_loss_guide/dataset_size,}, iteration + 1)
        writer.add_scalars('loss_norm1', {'loss_norm1'+str(fold):total_loss_norm1/dataset_size,}, iteration + 1)
        writer.add_scalars('loss_mse', {'loss_mse'+str(fold):total_loss_mse/dataset_size,}, iteration + 1)
        writer.add_scalars('loss_fmse', {'loss_fmse'+str(fold):total_loss_fmse/dataset_size,}, iteration + 1)
        writer.add_scalars('loss_g_ad', {'loss_g_ad'+str(fold):total_loss_g_ad/dataset_size,}, iteration + 1)
        writer.add_scalars('loss_d_ad', {'loss_d_ad'+str(fold):total_loss_d_ad/dataset_size,}, iteration + 1)
        writer.add_scalars('loss_ssim', {'loss_ssim'+str(fold):total_loss_ssim/dataset_size,}, iteration + 1)
        writer.add_scalars('loss_mae', {'loss_mae'+str(fold):total_loss_mae/dataset_size,}, iteration + 1)
        writer.add_scalars('train_acc', {'train_acc' + str(fold): train_acc, }, iteration+1)
        writer.add_scalars('train_auc', {'train_auc' + str(fold): train_auc, }, iteration+1)
        print("indices:",torch.sort(w_sl2[:,0], descending=True)[1].cpu().numpy().tolist()[0:20])
        if (iteration+1) >= 30 and (iteration+1)%5==0:
            torch.save(G2.state_dict(), os.path.join(MODEL_PATH, 'Fold_'+str(fold)+'Epoch{}_G2.pth'.format(iteration + 1)))
            torch.save(w_sl2, os.path.join(MODEL_PATH, 'Fold_'+str(fold)+'Epoch{}_w_sl2.pth'.format(iteration + 1)))
         ###############test######################
        
        T_ = 0
        F_ = 0
        G.eval()
        G2.eval() 
        G3.eval()
        D.eval()  
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

        writer.add_scalars('test_acc', {'test_acc' + str(fold): test_acc, }, iteration+1)
        writer.add_scalars('test_auc', {'test_auc' + str(fold): test_auc, }, iteration+1)
        writer.add_scalars('test_sen', {'test_sen' + str(fold): test_sen, }, iteration+1)
        writer.add_scalars('test_spe', {'test_spe' + str(fold): test_spe, }, iteration+1)

        if  opt.use_sparse2 and round(train_acc,2) >= acc_th and (iteration+1)>=10:
            w_sl2.requires_grad = True
            sparse_flag = 1
