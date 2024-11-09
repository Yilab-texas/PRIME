import torch
import torch.nn as nn
import torch.nn.functional as F 

class Self_Attention(nn.Module):
    
    def __init__(self, output_dim):
        super(Self_Attention, self).__init__()
        self.output_dim = output_dim
        self.kernel = nn.Parameter(torch.Tensor(3, output_dim, output_dim))
        nn.init.uniform_(self.kernel, -0.05, 0.05)
    
    def forward(self, x):
        b = int(x.size(0))
        d = int(x.size(1))
        d_2 = int(d / 32)
        x = x.reshape(b, d_2, 32)
        WQ = torch.matmul(x, self.kernel[0])
        WK = torch.matmul(x, self.kernel[1])
        WV = torch.matmul(x, self.kernel[2])

        QK = torch.matmul(WQ, WK.transpose(1, 2))
        QK = QK / (self.output_dim**0.5)

        QK = F.softmax(QK, dim=-1)

        V = torch.matmul(QK, WV)

        V = V.reshape(b, d)
        return V

class SdnnModel(nn.Module):
    def __init__(self, in_features=573, out_features=32, dropout_p=0.1, use_esm=False):
        super(SdnnModel, self).__init__()
        in_features = in_features
        out_features = out_features
        if use_esm:
            esm_merge_dim = out_features
        else:
            esm_merge_dim = 0


        self.channel_1 = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_p),
            Self_Attention(out_features),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, out_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_features),
            nn.Dropout(p=dropout_p)
        )

        self.channel_2 = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_p),
            Self_Attention(out_features),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, out_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_features),
            nn.Dropout(p=dropout_p)
        )

        self.merged = nn.Linear(2*out_features, out_features)
        self.prediction_module = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(inplace=True),
            Self_Attention(out_features),
            nn.Linear(out_features, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 2),
            # nn.Sigmoid()
            # nn.ReLU(inplace=True),
        )

        self.esm_model = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            # Self_Attention(out_features),
            nn.Linear(512, out_features),
            nn.ReLU(inplace=True),
        )

        self.discrim_model = nn.Sequential(
            nn.Linear(out_features*2 + esm_merge_dim, out_features),
            # nn.ReLU(inplace=True),
            # # Self_Attention(out_features),
            # nn.Linear(out_features, 2),
            # nn.ReLU(inplace=True),
        )

        self.discrim_model_pred = nn.Sequential(
            # nn.Linear(out_features*2 + esm_merge_dim, out_features),
            nn.ReLU(inplace=True),
            # Self_Attention(out_features),
            nn.Linear(out_features, 2),
            nn.ReLU(inplace=True),
        )

    def forward(self,x1,x2):
        out1 = self.channel_1(x1)
        out2 = self.channel_2(x2)
        merged_out = torch.cat((out1, out2), dim=1)
        merged_out = self.merged(merged_out)
        output = self.prediction_module(merged_out)
        return output, merged_out  

    def discriminator(self, A_B_merged, A_2_B_merged):
        merged_out = torch.cat((A_B_merged, A_2_B_merged), dim=1)
        merged_out = self.discrim_model(merged_out)
        out = self.discrim_model_pred(merged_out)
        return out 

    def discriminator_esm(self, A_B_merged, A_2_B_merged, esm_feat):
        esm_merged = self.esm_model(esm_feat)
        merged_out = torch.cat((A_B_merged, A_2_B_merged, esm_merged), dim=1)
        # out = self.discrim_model(merged_out)
        merged_out = self.discrim_model(merged_out)
        out = self.discrim_model_pred(merged_out)
        return out, esm_merged, merged_out
