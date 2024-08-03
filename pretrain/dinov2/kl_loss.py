import torch.nn as nn
import torch.nn.functional as F
import math


class KLDiv(nn.Module):
    def __init__(self, temperature=1.0):
        super(KLDiv, self).__init__()
        self.temperature = temperature

    def forward(self, z_s, z_t, **kwargs):
        log_pred_student = F.log_softmax(z_s / self.temperature, dim=-1)
        pred_teacher = F.softmax(z_t / self.temperature, dim=-1)

        kd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')*self.temperature*self.temperature
        
        return kd_loss


class Cosine(nn.Module):
    def __init__(self, dim=-1,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cos = nn.CosineSimilarity(dim=dim)
    
    def forward(self, student_out, teacher_out):
        loss = 1 - self.cos(student_out, teacher_out).mean()
        return loss


class FeatureLoss(nn.Module):
    def __init__(self, alpha=0.9, beta=0.1, dim=-1) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cos_fn = Cosine(dim=dim)
        self.smoothl1 = nn.SmoothL1Loss()
    
    def forward(self, student_out, teacher_out):
        """
        student_out: (B, N1, C)
        teacher_out: (B, N2, C)
        """
        
        N1 = student_out.shape[1]
        B, N2, C = teacher_out.shape
        l1 = int(math.sqrt(N1))
        l2 = int(math.sqrt(N2))
        teacher_out = teacher_out.permute(0, 2, 1).view(B, C, l2, l2)
        teacher_out = F.interpolate(teacher_out, size=(l1, l1), mode='bicubic')
        teacher_out = teacher_out.view(B, C, -1).permute(0, 2, 1)
        
        cos = self.cos_fn(student_out, teacher_out) * self.alpha
        smoothl1 = self.smoothl1(student_out, teacher_out) * self.beta
        return cos + smoothl1


def get_kd_loss(name):
    if name.lower() == 'kldiv':
        loss = KLDiv(temperature=1.0)
        
    elif name.lower() == 'radio':
        loss = Cosine()
    else:
        raise NotImplementedError
    return loss


if __name__ =='__main__':
    import torch
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    loss_fn = KLDiv()
    loss = loss_fn(x, y)
    print(loss)
    