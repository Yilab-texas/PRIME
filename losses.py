import torch
import torch.nn as nn

class EditLoss(nn.Module):
    def __init__(self, func='mean'):
        super(EditLoss, self).__init__()
        self.func = func

    def forward(self, input_tensor):
        num_elements = input_tensor.size(0)
        expanded_tensor = input_tensor.unsqueeze(0).expand(num_elements, -1,)
        mse_loss_matrix = nn.functional.mse_loss(expanded_tensor, expanded_tensor.transpose(0, 1), reduction='none')

        if self.func == 'mean':
            mse_loss = mse_loss_matrix.mean()
        elif self.func == 'sum':
            mse_loss = mse_loss_matrix.sum()
        return mse_loss