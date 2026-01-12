import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion_Level_MetricLearner(nn.Module):
    def __init__(self,config):
        super(Fusion_Level_MetricLearner, self).__init__()
        in_channel = config['in_channel']
        out_channel = config['out_channel']
        self.no_use = nn.Linear(1,1)
     
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        bs = x.shape[0]
        x = F.normalize(x.view(bs,-1), dim=1)
        return x

class Agent_Level_MetricLearner(nn.Module):
    def __init__(self,config):
        super(Agent_Level_MetricLearner, self).__init__()
        in_channel = config['in_channel']
        out_channel = config['out_channel']
        self.no_use = nn.Linear(1,1)
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        bs = x.shape[0]
        x = F.normalize(x.view(bs,-1), dim=1)
        return x