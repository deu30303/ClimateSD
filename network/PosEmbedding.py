import torch
import torch.nn as nn
import torch.nn.functional as F
from network.mynn import initialize_embedding
import numpy as np

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''
    def cal_angle(position, hid_idx):
        if d_hid > 50:
            cycle = 10
        elif d_hid > 5:
            cycle = 100
        else:
            cycle = 10000
        cycle = 10 if d_hid > 50 else 100
        return position / np.power(cycle, 2 * (hid_idx // 2) / d_hid)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)



class PosEncoding1D(nn.Module):
    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEncoding1D, self).__init__()
        print("use PosEncoding1D")
        self.sel_index = torch.tensor([0])
        pos_enc = (get_sinusoid_encoding_table((128//pos_rfactor)+1, dim) + 1)
        self.pos_layer = nn.Embedding.from_pretrained(embeddings=pos_enc, freeze=True)
        self.pos_noise = pos_noise
        self.noise_clamp = 16 // pos_rfactor # 4: 4, 8: 2, 16: 1

        self.pos_rfactor = pos_rfactor
        if pos_noise > 0.0:
            self.min = 0.0 #torch.tensor([0]).cuda()
            self.max = 128//pos_rfactor #torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))

    def forward(self, x, pos, return_posmap=False):
        pos_h, _ = pos # B X H X W
        pos_h = pos_h//self.pos_rfactor
        pos_h = pos_h.unsqueeze(0)
        pos_h = pos_h.index_select(2, self.sel_index).unsqueeze(1).squeeze(3) # B X 1 X H
        pos_h = nn.functional.interpolate(pos_h.float(), size=x.shape[2], mode='nearest').long() # B X 1 X 48

        if self.training is True and self.pos_noise > 0.0:
            pos_h = pos_h + torch.clamp((self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long(), 
                            min=-self.noise_clamp, max=self.noise_clamp)
            pos_h = torch.clamp(pos_h, min=self.min, max=self.max)

        pos_h = self.pos_layer(pos_h.cuda()).transpose(1,3).squeeze(3)   # B X 1 X 48 X 80 > B X 80 X 48 X 1 
        
        x = x + pos_h
        if return_posmap:
            return x, self.pos_layer.weight # 33 X 80
        return x
    
class PosEncoding1D_W(nn.Module):
    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEncoding1D_W, self).__init__()
        print("use PosEncoding1D_W")
        self.sel_index = torch.tensor([0])
        pos_enc = (get_sinusoid_encoding_table((128//pos_rfactor)+1, dim) + 1)
        self.pos_layer = nn.Embedding.from_pretrained(embeddings=pos_enc, freeze=True)
        self.pos_noise = pos_noise
        self.noise_clamp = 16 // pos_rfactor # 4: 4, 8: 2, 16: 1

        self.pos_rfactor = pos_rfactor
        if pos_noise > 0.0:
            self.min = 0.0 #torch.tensor([0]).cuda()
            self.max = 128//pos_rfactor #torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))

    def forward(self, x, pos, return_posmap=False):
        _, pos_w = pos # B X H X W
        pos_w = pos_w//self.pos_rfactor
        pos_w = pos_w.unsqueeze(0)
        pos_w = pos_w.index_select(2, self.sel_index).unsqueeze(1).squeeze(3) # B X 1 X H
        pos_w = nn.functional.interpolate(pos_w.float(), size=x.shape[2], mode='nearest').long() # B X 1 X 48

        if self.training is True and self.pos_noise > 0.0:
            pos_w = pos_w + torch.clamp((self.noise.sample(pos_w.shape).squeeze(3).cuda()//1).long(), 
                            min=-self.noise_clamp, max=self.noise_clamp)
            pos_w = torch.clamp(pos_w, min=self.min, max=self.max)

        pos_w = self.pos_layer(pos_w.cuda()).transpose(1,3).squeeze(3)   # B X 1 X 48 X 80 > B X 80 X 48 X 1 
        
        x = x + pos_w
        if return_posmap:
            return x, self.pos_layer.weight # 33 X 80
        return x
    
    
class PosEncoding2D(nn.Module):
    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEncoding2D, self).__init__()
        print("use PosEncoding2D")
        self.sel_index = torch.tensor([0]).cuda()
        pos_enc = (get_sinusoid_encoding_table((128//pos_rfactor)+1, dim) + 1)
        self.pos_layer_h = nn.Embedding.from_pretrained(embeddings=pos_enc, freeze=True)
        self.pos_layer_w = nn.Embedding.from_pretrained(embeddings=pos_enc, freeze=True)
        self.pos_noise = pos_noise
        self.noise_clamp = 16 // pos_rfactor
        self.pos_rfactor = pos_rfactor
        if pos_noise > 0.0:
            self.min = 0.0 #torch.tensor([0]).cuda()
            self.max = 128//pos_rfactor #torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))
    def forward(self, x, pos, return_posmap=False):
        pos_h, pos_w = pos # B X H X W
        pos_h = pos_h//self.pos_rfactor
        pos_h = pos_h.unsqueeze(0).cuda()
        pos_h = pos_h.index_select(2, self.sel_index.cuda()).unsqueeze(1).squeeze(3) # B X 1 X H
        pos_h = nn.functional.interpolate(pos_h.float(), size=x.shape[2], mode='nearest').long()
        pos_w = pos_w//self.pos_rfactor
        pos_w = pos_w.unsqueeze(0).cuda()
        pos_w = pos_w.index_select(2, self.sel_index.cuda()).unsqueeze(2).squeeze(3) # B X W X 1
        pos_w = nn.functional.interpolate(pos_h.float(), size=x.shape[2], mode='nearest').long()
        if self.training is True and self.pos_noise > 0.0:
            pos_h = pos_h + torch.clamp((self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long(),
                            min=-self.noise_clamp, max=self.noise_clamp)
            pos_h = torch.clamp(pos_h, min=self.min, max=self.max)
            pos_w = pos_w + torch.clamp((self.noise.sample(pos_w.shape).squeeze(3).cuda()//1).long(),
                            min=-self.noise_clamp, max=self.noise_clamp)
            pos_w = torch.clamp(pos_w, min=self.min, max=self.max)
        pos_h = self.pos_layer_h(pos_h).transpose(1,3).squeeze(3).squeeze(0)   # B X 1 X 48 X 80 > B X 80 X 48 X 1
        pos_w = self.pos_layer_w(pos_w).transpose(1,3).squeeze(3).squeeze(0)   # B X 1 X 48 X 80 > B X 80 X 48 X 1
        pos = torch.matmul(pos_h.unsqueeze(2), pos_w.unsqueeze(1)).unsqueeze(0)
        x = x + pos
        if return_posmap:
            return x, self.pos_layer.weight # 33 X 80
        return x

    

class PosEmbedding1D(nn.Module):
    
    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEmbedding1D, self).__init__()
        print("use PosEmbedding1D")
        self.sel_index = torch.tensor([0]).cuda()
        self.pos_layer = nn.Embedding((128//pos_rfactor)+1, dim)
        initialize_embedding(self.pos_layer)
        self.pos_noise = pos_noise
        self.pos_rfactor = pos_rfactor
        self.noise_clamp = 16 // pos_rfactor # 4: 4, 8: 2, 16: 1

        if pos_noise > 0.0:
            self.min = 0.0 #torch.tensor([0]).cuda()
            self.max = 128//pos_rfactor #torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))

    def forward(self, x, pos, return_posmap=False):
        pos_h, _ = pos # B X H X W
        pos_h = pos_h//self.pos_rfactor
        pos_h = pos_h.index_select(2, self.sel_index).unsqueeze(1).squeeze(3).cuda() # B X 1 X H
        pos_h = nn.functional.interpolate(pos_h.float(), size=x.shape[2], mode='nearest').long() # B X 1 X 48

        if self.training is True and self.pos_noise > 0.0:
            #pos_h = pos_h + (self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long()
            pos_h = pos_h + torch.clamp((self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long(),
                            min=-self.noise_clamp, max=self.noise_clamp)
            pos_h = torch.clamp(pos_h, min=self.min, max=self.max)

        pos_h = self.pos_layer(pos_h).transpose(1,3).squeeze(3)   # B X 1 X 48 X 80 > B X 80 X 48 X 1 
        x = x + pos_h
        if return_posmap:
            return x, self.pos_layer.weight # 33 X 80
        return x
    
class PosEmbedding1D_W(nn.Module):
    
    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEmbedding1D_W, self).__init__()
        print("use PosEmbedding1D_W")
        self.sel_index = torch.tensor([0]).cuda()
        self.pos_layer = nn.Embedding((128//pos_rfactor)+1, dim)
        initialize_embedding(self.pos_layer)
        self.pos_noise = pos_noise
        self.pos_rfactor = pos_rfactor
        self.noise_clamp = 16 // pos_rfactor # 4: 4, 8: 2, 16: 1

        if pos_noise > 0.0:
            self.min = 0.0 #torch.tensor([0]).cuda()
            self.max = 128//pos_rfactor #torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))

    def forward(self, x, pos, return_posmap=False):
        _, pos_w = pos # B X H X W
        pos_w = pos_w//self.pos_rfactor
        pos_w = pos_w.index_select(2, self.sel_index).unsqueeze(1).squeeze(3).cuda() # B X 1 X H
        pos_w = nn.functional.interpolate(pos_w.float(), size=x.shape[2], mode='nearest').long() # B X 1 X 48

        if self.training is True and self.pos_noise > 0.0:
            #pos_h = pos_h + (self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long()
            pos_w = pos_w + torch.clamp((self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long(),
                            min=-self.noise_clamp, max=self.noise_clamp)
            pos_w = torch.clamp(pos_h, min=self.min, max=self.max)

        pos_w = self.pos_layer(pos_w).transpose(1,3).squeeze(3)   # B X 1 X 48 X 80 > B X 80 X 48 X 1 
        x = x + pos_w
        if return_posmap:
            return x, self.pos_layer.weight # 33 X 80
        return x
    
class PosEmbedding2D(nn.Module):
    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEmbedding2D, self).__init__()
        print("use PosEmbedding2D")
        self.sel_index = torch.tensor([0]).cuda()
        pos_enc = (get_sinusoid_encoding_table((128//pos_rfactor)+1, dim) + 1)
        self.pos_layer_h = nn.Embedding((128//pos_rfactor)+1, dim)
        self.pos_layer_w = nn.Embedding((128//pos_rfactor)+1, dim)
        self.pos_noise = pos_noise
        self.noise_clamp = 16 // pos_rfactor
        self.pos_rfactor = pos_rfactor
        if pos_noise > 0.0:
            self.min = 0.0 #torch.tensor([0]).cuda()
            self.max = 128//pos_rfactor #torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))
    def forward(self, x, pos, return_posmap=False):
        pos_h, pos_w = pos # B X H X W
        pos_h = pos_h//self.pos_rfactor
        pos_h = pos_h.unsqueeze(0).cuda()
        pos_h = pos_h.index_select(2, self.sel_index.cuda()).unsqueeze(1).squeeze(3) # B X 1 X H
        
        pos_h = nn.functional.interpolate(pos_h.float(), size=x.shape[2], mode='nearest').long()
        
        
        pos_w = pos_w//self.pos_rfactor
        pos_w = pos_w.unsqueeze(0).cuda()
        pos_w = pos_w.index_select(2, self.sel_index.cuda()).unsqueeze(2).squeeze(3) # B X W X 1
        pos_w = nn.functional.interpolate(pos_h.float(), size=x.shape[2], mode='nearest').long()
        if self.training is True and self.pos_noise > 0.0:
            pos_h = pos_h + torch.clamp((self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long(),
                            min=-self.noise_clamp, max=self.noise_clamp)
            pos_h = torch.clamp(pos_h, min=self.min, max=self.max)
            pos_w = pos_w + torch.clamp((self.noise.sample(pos_w.shape).squeeze(3).cuda()//1).long(),
                            min=-self.noise_clamp, max=self.noise_clamp)
            pos_w = torch.clamp(pos_w, min=self.min, max=self.max)
        pos_h = self.pos_layer_h(pos_h).transpose(1,3).squeeze(3).squeeze(0)   # B X 1 X 48 X 80 > B X 80 X 48 X 1
        pos_w = self.pos_layer_w(pos_w).transpose(1,3).squeeze(3).squeeze(0)   # B X 1 X 48 X 80 > B X 80 X 48 X 1
        pos = torch.matmul(pos_h.unsqueeze(2), pos_w.unsqueeze(1)).unsqueeze(0)
        x = x + pos
        if return_posmap:
            return x, self.pos_layer.weight # 33 X 80
        return x