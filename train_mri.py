import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix

from tensorboardX import SummaryWriter
from model.transformer import S_Transformer
from model.ScaleDense import ScaleDense_VAE,ScaleDense_Dis
from utils.mri_gene_dataset import MRIandGenedataset
from options.train_options import TrainOptions
from utils.base_function import GANLoss, criterion_psnr
from utils.SSIM_loss import SSIM
from utils.base_function import _freeze,_unfreeze

cuda = torch.cuda.is_available()

opt = TrainOptions().parse()
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
LOG_PATH =  f"./logs/log_s-mri_{alpha_mse}_{alpha_guide}_{opt.alpha_sparse}_{opt.mri_th}"
os.system("mkdir -p {}".format(MODEL_PATH))
os.system("mkdir -p {}".format(LOG_PATH))

# '''!!!!!!!!!!!!!!!!!!manually!!!!!!!!!!!!!!!!!!!!!!!!'''
TRAIN_BATCH_SIZE = 8#32
TEST_BATCH_SIZE = 8#32
lr = 1e-4
EPOCH = 150
WORKERS = 8
WIDTH = 32
NUM = 1200
NUM2 = 441

writer = SummaryWriter(logdir=LOG_PATH, comment='Gene2MRI')
criterion_bce = nn.BCELoss().cuda()
criterion_l1 = nn.L1Loss().cuda()
criterion_mse = nn.MSELoss().cuda()
criterion_ssim = SSIM().cuda()
criterion = nn.CrossEntropyLoss().cuda()
gan_loss = GANLoss('lsgan')

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for fold in range(5):
    # if fold != 4:
    #     continue
    # load train data
    dataset_train = MRIandGenedataset(i=fold,opt="train")
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                                    num_workers=WORKERS)
    dataset_test = MRIandGenedataset(i=fold,opt="test")
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=True,
                                                    num_workers=WORKERS)
    dataset_size = len(data_loader_train)
    dataset_size_test = len(data_loader_test)
    print("train size: ", dataset_size)
    print("test size: ", dataset_size_test)
    G = ScaleDense_VAE().cuda()
    G3 = S_Transformer(opt).cuda()   
    D = ScaleDense_Dis().cuda()  

    lr_changed = 0
    sparse_flag = 0
    
    w_sl = torch.ones([NUM,1])
    w_sl = w_sl.cuda()
    w_sl.requires_grad = False
    w_sl2 = torch.ones([NUM2,1])
    w_sl2 = w_sl2.cuda()
    w_sl2.requires_grad = False
    for p in G.parameters():
        p.requires_grad = True
    for p in G3.parameters():
        p.requires_grad = True
    for p in D.parameters():
        p.requires_grad = True
    g_optimizer = optim.AdamW([{'params': G.parameters()},{'params': G3.parameters()}], lr=lr) 
    d_optimizer = optim.AdamW([{'params': D.parameters()}], lr=lr*0.1) 
    w_optimizer = optim.AdamW([{'params': w_sl}], lr=5e-3) 
    
    cover_flag = 0
    for iteration in range(EPOCH):
        ###############train######################
        print("learning_rate:",g_optimizer.param_groups[0]['lr'])
        print("learning_rate:",w_optimizer.param_groups[0]['lr'])
        print(iteration + 1)
        G.train()
        G3.train()    
        D.train()  
        total_loss = 0
        total_loss_mse = 0 
        total_loss_norm1 = 0 
        total_loss_guide = 0
        total_loss_d_ad = 0
        total_loss_g_ad = 0
        
        for index,train_data in enumerate(data_loader_train):
            fid, label, integer_encoded,age_sex,input = train_data #1500 1536
            
            label = label.cuda()
            B, L = integer_encoded.shape
            
            input = input.cuda()
            age_sex = age_sex.cuda()
            snp = integer_encoded.cuda()

            feature,x_list = G(input) 
            x_ = x_list[-1]
            feature = feature.view(B,opt.hidden_size,-1).permute([0,2,1])
            
            y, gate, mri_feature = G3(w_sl = w_sl,inputs_embeds=feature,width = WIDTH,batch = B)  

            loss_guide = criterion(y, label)
            loss_norm1 = torch.norm(gate,p=1) 
            loss_mse =  criterion_mse(input,x_)

            _unfreeze(D)
            D_real = D(input.detach())
            D_fake = D(x_.detach())
            loss_d_ad = (gan_loss(D_real, True, True) + gan_loss(D_fake, False, True))*0.5
            loss_d  = loss_d_ad
            d_optimizer.zero_grad()
            loss_d.backward()
            d_optimizer.step()
            _freeze(D)   
            D_fake = D(x_)
            loss_g_ad = gan_loss(D_fake, True, False)  
            
            ##############################################   
            g_optimizer.zero_grad()   
            w_optimizer.zero_grad()
            if sparse_flag:
                loss_g = alpha_guide *  loss_guide + (10**(opt.alpha_sparse)) * F.relu(loss_norm1-opt.mri_th)  + alpha_mse * loss_mse + loss_g_ad
            else:
                loss_g = alpha_guide *  loss_guide + alpha_mse * loss_mse + loss_g_ad
           
            loss_g.backward()
            g_optimizer.step()            
            w_optimizer.step()     
            
            if not lr_changed and round(loss_norm1.item()) <= opt.mri_th:
                adjust_learning_rate(w_optimizer, 1e-3)
                lr_changed = 1

            total_loss_mse += loss_mse.item()
            total_loss_norm1 += loss_norm1.item()
            total_loss_guide += loss_guide.item()
            total_loss_d_ad += loss_d_ad.item()
            total_loss_g_ad += loss_g_ad.item()

        T_ = 0
        F_ = 0
        G.eval()
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
                feature = G(input,out_rec=False)
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
                
        train_acc = T_  / (T_ + F_)
        train_auc = roc_auc_score(label_all,prob_all)

        print("loss_guide:",total_loss_guide/dataset_size)
        print("loss_norm1:",total_loss_norm1/dataset_size)
        print("loss_mse:",total_loss_mse/dataset_size)
        print("loss_g_ad:",total_loss_g_ad/dataset_size)
        print("loss_d:",total_loss_d_ad/dataset_size)
        print("train_acc:",train_acc)
        print("train_auc:",train_auc)
        
        writer.add_scalars('loss_guide', {'loss_guide'+str(fold):total_loss_guide/dataset_size,}, iteration + 1)
        writer.add_scalars('loss_norm1', {'loss_norm1'+str(fold):total_loss_norm1/dataset_size,}, iteration + 1)
        writer.add_scalars('loss_mse', {'loss_mse'+str(fold):total_loss_mse/dataset_size,}, iteration + 1)
        writer.add_scalars('loss_g_ad', {'loss_g_ad'+str(fold):total_loss_g_ad/dataset_size,}, iteration + 1)
        writer.add_scalars('loss_d_ad', {'loss_d_ad'+str(fold):total_loss_d_ad/dataset_size,}, iteration + 1)
        writer.add_scalars('train_acc', {'train_acc' + str(fold): train_acc, }, iteration+1)
        writer.add_scalars('train_auc', {'train_auc' + str(fold): train_auc, }, iteration+1)
        if (iteration+1)%10==0:
            torch.save(G.state_dict(), os.path.join(MODEL_PATH, 'Fold_'+str(fold)+'Epoch{}_G.pth'.format(iteration + 1)))
            torch.save(G3.state_dict(), os.path.join(MODEL_PATH, 'Fold_'+str(fold)+'Epoch{}_G3.pth'.format(iteration + 1)))
            torch.save(w_sl, os.path.join(MODEL_PATH, 'Fold_'+str(fold)+'Epoch{}_w_sl.pth'.format(iteration + 1)))
       
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

        writer.add_scalars('test_acc', {'test_acc' + str(fold): test_acc, }, iteration+1)
        writer.add_scalars('test_auc', {'test_auc' + str(fold): test_auc, }, iteration+1)
        writer.add_scalars('test_sen', {'test_sen' + str(fold): test_sen, }, iteration+1)
        writer.add_scalars('test_spe', {'test_spe' + str(fold): test_spe, }, iteration+1)
        
        test_ssim = np.mean(np.array(ssim_all))
        test_psnr = np.mean(np.array(psnr_all))
        print("test_ssim:", test_ssim)
        print("test_psnr:", test_psnr)

        writer.add_scalars('test_ssim', {'test_ssim' + str(fold): test_ssim, }, iteration+1)
        writer.add_scalars('test_psnr', {'test_psnr' + str(fold): test_psnr, }, iteration+1)

        if opt.use_sparse and iteration>=100:
            w_sl.requires_grad = True
            sparse_flag = 1 
        
        if (iteration+1)%10==0 and  cover_flag:
            break
        if (iteration+1)%10==0 and round(loss_norm1.item()) <= opt.mri_th:
            cover_flag = 1