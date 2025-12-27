import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion_Level_MetricLearner(nn.Module):
    def __init__(self,config):
        super(Fusion_Level_MetricLearner, self).__init__()
        in_channel = config['in_channel']
        out_channel = config['out_channel']
        self.setting = config['setting']
        self.no_use = nn.Linear(1,1)
        if self.setting == 0:
            self.avgpool = nn.AvgPool2d(kernel_size=(100,352),stride=(100,352))
            self.fc1 = nn.Linear(in_channel, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, out_channel)
            
        if self.setting == 1:
            self.avgpool = nn.AvgPool2d(kernel_size=(100,352),stride=(100,352))
            
        if self.setting == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=(100,352),stride=(100,352))
            
        if self.setting == 3:
            self.avgpool = nn.AvgPool2d(kernel_size=(50,176),stride=(50,176))
           
        if self.setting == 4:
            self.maxpool = nn.MaxPool2d(kernel_size=(50,176),stride=(50,176))
            
        if self.setting == 5:
            self.avgpool = nn.AvgPool2d(kernel_size=(100,352),stride=(100,352))
            self.maxpool = nn.MaxPool2d(kernel_size=(100,352),stride=(100,352))
            
        if self.setting == 6:
            self.maxpool = nn.MaxPool2d(kernel_size=(50,176),stride=(50,176))
            self.avgpool = nn.AvgPool2d(kernel_size=(50,176),stride=(50,176))
     
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        bs = x.shape[0]
        if self.setting == 0:
            x = self.avgpool(x)
            x = x.view(bs,-1)
            x = self.fc2(self.relu(self.fc1(x)))
            x = F.normalize(x, dim=1)
        if self.setting == 1:
            x = self.avgpool(x)
            x = x.view(bs,-1)
            x = F.normalize(x, dim=1)
        if self.setting == 2:
            x = self.maxpool(x)
            x = x.view(bs,-1)
            x = F.normalize(x, dim=1)
        if self.setting == 3:
            x = self.avgpool(x)
            x = x.view(bs,-1)
            x = F.normalize(x, dim=1)
        if self.setting == 4:
            x = self.maxpool(x)
            x = x.view(bs,-1)
            x = F.normalize(x, dim=1)
        if self.setting == 5:
            x_max = self.maxpool(x)
            x_avg = self.avgpool(x)
            x_max = x_max.view(bs,-1)
            x_avg = x_avg.view(bs,-1)
            x = torch.cat((x_max,x_avg),dim=1)
            x = F.normalize(x, dim=1)
        if self.setting == 6:
            x_max = self.maxpool(x)
            x_avg = self.avgpool(x)
            x_max = x_max.view(bs,-1)
            x_avg = x_avg.view(bs,-1)
            x = torch.cat((x_max,x_avg),dim=1)
            x = F.normalize(x, dim=1)
        if self.setting == 7:
            x = F.normalize(x.view(bs,-1), dim=1)
        return x

class Agent_Level_MetricLearner(nn.Module):
    def __init__(self,config):
        super(Agent_Level_MetricLearner, self).__init__()
        in_channel = config['in_channel']
        out_channel = config['out_channel']
        self.setting = config['setting']
        self.no_use = nn.Linear(1,1)
        if self.setting == 0:
            self.avgpool = nn.AvgPool2d(kernel_size=(100,352),stride=(100,352))
            self.fc1 = nn.Linear(in_channel, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, out_channel)
            
        if self.setting == 1:
            self.avgpool = nn.AvgPool2d(kernel_size=(100,352),stride=(100,352))
            
        if self.setting == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=(100,352),stride=(100,352))
            
        if self.setting == 3:
            self.avgpool = nn.AvgPool2d(kernel_size=(50,176),stride=(50,176))
           
        if self.setting == 4:
            self.maxpool = nn.MaxPool2d(kernel_size=(50,176),stride=(50,176))
            
        if self.setting == 5:
            self.avgpool = nn.AvgPool2d(kernel_size=(100,352),stride=(100,352))
            self.maxpool = nn.MaxPool2d(kernel_size=(100,352),stride=(100,352))
            
        if self.setting == 6:
            self.maxpool = nn.MaxPool2d(kernel_size=(50,176),stride=(50,176))
            self.avgpool = nn.AvgPool2d(kernel_size=(50,176),stride=(50,176))
     
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        bs = x.shape[0]
        if self.setting == 0:
            x = self.avgpool(x)
            x = x.view(bs,-1)
            x = self.fc2(self.relu(self.fc1(x)))
            x = F.normalize(x, dim=1)
        if self.setting == 1:
            x = self.avgpool(x)
            x = x.view(bs,-1)
            x = F.normalize(x, dim=1)
        if self.setting == 2:
            x = self.maxpool(x)
            x = x.view(bs,-1)
            x = F.normalize(x, dim=1)
        if self.setting == 3:
            x = self.avgpool(x)
            x = x.view(bs,-1)
            x = F.normalize(x, dim=1)
        if self.setting == 4:
            x = self.maxpool(x)
            x = x.view(bs,-1)
            x = F.normalize(x, dim=1)
        if self.setting == 5:
            x_max = self.maxpool(x)
            x_avg = self.avgpool(x)
            x_max = x_max.view(bs,-1)
            x_avg = x_avg.view(bs,-1)
            x = torch.cat((x_max,x_avg),dim=1)
            x = F.normalize(x, dim=1)
        if self.setting == 6:
            x_max = self.maxpool(x)
            x_avg = self.avgpool(x)
            x_max = x_max.view(bs,-1)
            x_avg = x_avg.view(bs,-1)
            x = torch.cat((x_max,x_avg),dim=1)
            x = F.normalize(x, dim=1)
        if self.setting == 7:
            x = F.normalize(x.view(bs,-1), dim=1)
        return x