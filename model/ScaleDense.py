import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num/1e6, 'Trainable': trainable_num /1e6}

class SE_block(nn.Module):
    def __init__(self, inchannels, reduction = 16 ):
        super(SE_block,self).__init__()
        self.GAP = nn.AdaptiveAvgPool3d((1,1,1))
        self.FC1 = nn.Linear(inchannels,inchannels//reduction)
        self.FC2 = nn.Linear(inchannels//reduction,inchannels)

    def forward(self,x):
        model_input = x
        x = self.GAP(x)
        x = torch.reshape(x,(x.size(0),-1))
        x = self.FC1(x)
        x = nn.ReLU()(x)
        x = self.FC2(x)
        x = nn.Sigmoid()(x)
        x = x.view(x.size(0),x.size(1),1,1,1)
        return model_input * x

class AC_layer(nn.Module):
    def __init__(self,inchannels, outchannels):
        super(AC_layer,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv2 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,1,3),stride=1,padding=(0,0,1),bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv3 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,1,1),stride=1,padding=(1,0,0),bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv4 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,3,1),stride=1,padding=(0,1,0),bias=False),
            nn.BatchNorm3d(outchannels))
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return x1 + x2 + x3 + x4

class dense_layer(nn.Module):
    def __init__(self,inchannels,outchannels):

        super(dense_layer,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            AC_layer(inchannels,outchannels),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            AC_layer(outchannels,outchannels),       
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            SE_block(outchannels),
            nn.MaxPool3d(2,2),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,1,1),stride=1,padding=0,bias=False),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            nn.MaxPool3d(2,2),
        )
        #self.drop = nn.Dropout3d(0.1)

    def forward(self,x):
        #x = self.drop(x)
        new_features = self.block(x)
        x = F.max_pool3d(x,2)
        x = torch.cat([new_features,x], 1)
        #x = self.block(new_features) + self.block2(x)
        return x

class dense_layer2(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(dense_layer2,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            nn.MaxPool3d(2,2),
        )
    #(83,104,79) (41,52,39) (20,26,19) (10,13,9)
    #(88,105,85) (44,52,42) (22,26,21) (11,13,10)
    #(85,100,85) (42,50,42) (21,25,21) (10,12,10)
    #(80,100,85) (40,50,42) (20,25,21) (10,12,10)
    def forward(self,x):
        #print(x.shape)
        new_features = self.block(x) # (32,47,34)  (18,23,17) (9 11 8)
        x = F.max_pool3d(x,2)  #(42,52,39) (21,26,19) (10 13 9)
        x = torch.cat([new_features,x], 1)
        return x
    
class dense_layer2_2(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(dense_layer2_2,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=2, dilation=2, bias=False),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),   
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
        )

    def forward(self,x):
        new_features = self.block(x)
        x = torch.cat([new_features,x], 1)
        return x
    
class up_layer(nn.Module):
    def __init__(self,inchannels,outchannels,size):
        super(up_layer,self).__init__()
        self.pool =  nn.Upsample(size)#nn.UpsamplingNearest2d(scale_factor=2)
        self.block = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1, bias=False),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            self.pool,
            nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),   
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
        )
        self.bypass = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,1,1),stride=1,padding=0, bias=False),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            self.pool,
        )

    def forward(self,x):
        x = self.block(x) + self.bypass(x)
        return x
    
class ScaleDense(nn.Module):
    def __init__(self,nb_filter=16, nb_block=4, use_gender=False):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense,self).__init__()
        self.nb_block = nb_block
        self.use_gender = use_gender
        self.pre = nn.Sequential(
            nn.Conv3d(1,nb_filter,kernel_size=7,stride=1
                     ,padding=1,dilation=2),
            nn.ELU(),
            )
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.gap = nn.AdaptiveAvgPool3d((1,1,1))
        self.deep_fc = nn.Sequential(
            nn.Linear(last_channels,32,bias=True),
            nn.ELU(),
            )

        self.male_fc = nn.Sequential(
            nn.Linear(2,16,bias=True),
            nn.Linear(16,8,bias=True),
            nn.ELU(),
            )
        self.end_fc_with_gender = nn.Sequential(
            nn.Linear(40,16),
            nn.Linear(16,2),
            #nn.ReLU()
            )
        self.end_fc_without_gender = nn.Sequential(
            nn.Linear(32,16),
            nn.Linear(16,2),
            #nn.ReLU()
            )


    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):
            outchannels = nb_filter * pow(2,i+1)#inchannels * 2
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels

    def forward(self, x, male_input=None):
        x = self.pre(x)
        x = self.block(x)
        x = self.gap(x)
        x = torch.reshape(x,(x.size(0),-1))
        x = self.deep_fc(x)
        if self.use_gender:
            male = torch.reshape(male_input,(male_input.size(0),-1))
            male = self.male_fc(male)
            x = torch.cat([x,male.type_as(x)],1)
            x = self.end_fc_with_gender(x)
        else:
            x = self.end_fc_without_gender(x)
        return x

class ScaleDense2(nn.Module):
    def __init__(self,nb_filter=16, nb_block=4):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense2,self).__init__()
        self.nb_block = nb_block
        self.pre = nn.Sequential(
            nn.Conv3d(1,nb_filter,kernel_size=7,stride=1,padding=1,dilation=2),
            nn.ELU(),
            )
        #self.pool1 =  nn.MaxPool3d(3, stride=2)
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels,512,(1,1,1),stride=1,padding=0,bias=False),
            nn.BatchNorm3d(512),
            nn.ELU(),
            )

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):
            outchannels = inchannels * 2
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels

    def forward(self, x):
        x = self.pre(x)
        x = self.block(x)  
        feature = self.out(x)
        return feature

class ScaleDense3(nn.Module):
    def __init__(self,nb_filter=16, nb_block=3, nb_block2=2):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense3,self).__init__()
        self.nb_block = nb_block
        self.pre = nn.Sequential(
            nn.Conv3d(1,nb_filter,kernel_size=7,stride=1,padding=1,dilation=2),
            nn.ELU(),
            )
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.block2, last_channels2 = self._make_block_2(last_channels,nb_block2)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels2,512,(1,1,1),stride=1,padding=0,bias=False),
            nn.BatchNorm3d(512),
            nn.ELU(),
            )
        self.dropout1 =  nn.Dropout3d(0.2)
        self.dropout2 =  nn.Dropout3d(0.2)

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter * pow(2,i+1)
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def _make_block_2(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter
            blocks.append(dense_layer2_2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
     
    def forward(self, x):
        x = self.pre(x)
        x = self.block(x)
        x = self.dropout1(x)
        x = self.block2(x)
        x = self.dropout2(x)
        feature = self.out(x)
        return feature
    
class ScaleDense_VAE(nn.Module):
    def __init__(self,nb_filter=32, nb_block=3, nb_block2=2):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense_VAE,self).__init__()
        self.nb_block = nb_block
        # self.pre = nn.Sequential(
        #     nn.Conv3d(1,nb_filter,kernel_size=7,stride=1,padding=1,dilation=2),
        #     nn.ELU(),
        #     )
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.block2, last_channels2 = self._make_block_2(last_channels,nb_block2)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels2,512,(1,1,1),stride=1,padding=0,bias=False),
            nn.BatchNorm3d(512),
            nn.ELU(),
            )
        self.dropout1 =  nn.Dropout3d(0.2)
        self.dropout2 =  nn.Dropout3d(0.2)
        #(83,104,79) (41,52,39) (20,26,19) (10,13,9)
        #(85,100,85) (42,50,42) (21,25,21) (10,12,10)
        #(77,97,79) (38,48,39) (19,24,19) (9,12,9)
        #(80,100,85) (40,50,42) (20,25,21) (10,12,10)
        
        # self.up1 = nn.Sequential(
        #     nn.Conv3d(512,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False), 
        #     #nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),      
        #     nn.BatchNorm3d(nb_filter*4),
        #     nn.ELU(),            
        #     nn.Upsample((20,25,21)),
        #     nn.Conv3d(nb_filter*4,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False),     
        #     nn.BatchNorm3d(nb_filter*4),
        #     nn.ELU(),
        # )
        # self.up2 = nn.Sequential( 
        #     nn.Conv3d(nb_filter*4,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False), 
        #     #nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
        #     nn.BatchNorm3d(nb_filter*2),
        #     nn.ELU(),
        #     nn.Upsample((40,50,42)),
        #     nn.Conv3d(nb_filter*2,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False),     
        #     nn.BatchNorm3d(nb_filter*2),
        #     nn.ELU(),
        # )
        # self.up3 = nn.Sequential(
        #     nn.Conv3d(nb_filter*2,nb_filter,(3,3,3),stride=1,padding=1,bias=False), 
        #     #nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
        #     nn.BatchNorm3d(nb_filter),
        #     nn.ELU(),
        #     nn.Upsample((80,100,85)),
        #     nn.Conv3d(nb_filter,nb_filter,(3,3,3),stride=1,padding=1,bias=False),     
        #     nn.BatchNorm3d(nb_filter),
        #     nn.ELU(),
        # )
        
        self.up1 = nn.Sequential(
            nn.Upsample((20,25,21)),
            nn.Conv3d(512,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),      
            nn.BatchNorm3d(nb_filter*4),
            nn.ELU(),     
            nn.Conv3d(nb_filter*4,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter*4),
            nn.ELU(),
        )
        self.up2 = nn.Sequential( 
            nn.Upsample((40,50,42)),
            nn.Conv3d(nb_filter*4,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm3d(nb_filter*2),
            nn.ELU(),
            nn.Conv3d(nb_filter*2,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter*2),
            nn.ELU(),
        )
        self.up3 = nn.Sequential(
            nn.Upsample((80,100,85)),
            nn.Conv3d(nb_filter*2,nb_filter,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm3d(nb_filter),
            nn.ELU(),
            nn.Conv3d(nb_filter,nb_filter,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter),
            nn.ELU(),
        )
        
        self.out1 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*4,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out2 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*2,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out3 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter,1,(3,3,3),stride=1,padding=0,bias=False))

        self.out4 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter,1,(3,3,3),stride=1,padding=0,bias=False))
        
    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 1#nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter * pow(2,i)
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def _make_block_2(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter
            blocks.append(dense_layer2_2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
       
    def forward(self, x, out_rec=True):
        #84 104 78
        #x = self.pre(x) #74 94 68
        x = self.block(x)
        #x = self.dropout1(x)
        x = self.block2(x)
        #x = self.dropout2(x)
        feature = self.out(x)         
        feature = torch.tanh(feature)
        
        if out_rec: 
            x_1 =  self.up1(feature)
            x_2 =  self.up2(x_1)
            x_3 =  self.up3(x_2)
            #x_  = self.out4(x_3)
            #return feature, x_
            x_list = []
            x_list.append(self.out1(x_1))
            x_list.append(self.out2(x_2))
            x_list.append(self.out3(x_3))  
            return feature, x_list
        else:
            return feature
        
class res_layer(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(res_layer,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            nn.MaxPool3d(2,2),
        )
        self.bypass = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,1,1),stride=1,padding=0,bias=False),
            nn.MaxPool3d(2,2),
        )
    def forward(self,x):
        x = self.block(x) + self.bypass(x)
        return x
    
class ScaleDense_Dis(nn.Module):
    def __init__(self,nb_filter=16, nb_block=3, nb_block2=2):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense_Dis,self).__init__()
        self.nb_block = nb_block
        # self.pre = nn.Sequential(
        #     nn.Conv3d(1,nb_filter,kernel_size=7,stride=1,padding=1,dilation=2),
        #     nn.ELU(),
        #     )
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        #self.block2, last_channels2 = self._make_block_2(last_channels,nb_block2)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels,last_channels,(1,1,1),stride=1,padding=0,bias=False),
            nn.BatchNorm3d(last_channels),
            nn.ELU(),
            nn.Conv3d(last_channels,1,(1,1,1),stride=1,padding=0,bias=False),
            )

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 1#nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter * pow(2,i)
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    # def _make_block_2(self, nb_filter, nb_block):
    #     blocks = []
    #     inchannels = nb_filter
    #     for i in range(nb_block):         
    #         outchannels = nb_filter
    #         blocks.append(dense_layer2_2(inchannels,outchannels))
    #         inchannels = outchannels + inchannels
    #     return nn.Sequential(*blocks), inchannels
    
    def forward(self, x):
        #84 104 78
        #x = self.pre(x) #74 94 68
        x = self.block(x)
        #x = self.block2(x)
        x = self.out(x)
        # x = self.pool(x)[:,:,0,0,0]
        # x = self.fc(x)
        x = torch.sigmoid(x)
        return x

class ScaleDense_Dis2(nn.Module):
    def __init__(self,nb_filter=16, nb_block=3, nb_block2=2):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense_Dis2,self).__init__()
        self.nb_block = nb_block
        # self.pre = nn.Sequential(
        #     nn.Conv3d(1,nb_filter,kernel_size=7,stride=1,padding=1,dilation=2),
        #     nn.ELU(),
        #     )
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        #self.block2, last_channels2 = self._make_block_2(last_channels,nb_block2)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels,last_channels,(1,1,1),stride=1,padding=0,bias=False),
            nn.BatchNorm3d(last_channels),
            nn.ELU(),
            # nn.Conv3d(last_channels,1,(1,1,1),stride=1,padding=0,bias=False),
            )
        self.fc = nn.Linear(last_channels, 1)
        self.pooling = nn.AdaptiveAvgPool3d(((1,1,1)))

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 1#nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter * pow(2,i)
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    # def _make_block_2(self, nb_filter, nb_block):
    #     blocks = []
    #     inchannels = nb_filter
    #     for i in range(nb_block):         
    #         outchannels = nb_filter
    #         blocks.append(dense_layer2_2(inchannels,outchannels))
    #         inchannels = outchannels + inchannels
    #     return nn.Sequential(*blocks), inchannels
    
    def forward(self, x):
        #84 104 78
        #x = self.pre(x) #74 94 68
        x = self.block(x)
        #x = self.block2(x)
        x = self.out(x)
        x = self.pooling(x)[:,:,0,0,0]
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

class FC(nn.Module):
    def __init__(self,):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(FC,self).__init__()
        self.fc1 = nn.Linear(170,512)
        self.fc2 = nn.Linear(512,2)
        self.relu = nn.ReLU(0.2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
