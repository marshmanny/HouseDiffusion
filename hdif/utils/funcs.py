import torch
import torch.nn.functional as F

class DilatedMask:
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size
        
        cords_x = torch.arange(0, kernel_size).view(1, -1).expand(kernel_size, -1) - kernel_size // 2
        cords_y = cords_x.clone().permute(1, 0)
        self.kernel = torch.as_tensor((cords_x ** 2 + cords_y ** 2) <= (kernel_size // 2) ** 2, dtype=torch.float).view(1, 1, kernel_size, kernel_size).cuda()
        self.kernel /= self.kernel.sum()
    
    def __call__(self, mask):
        smooth_mask = F.conv2d(mask, self.kernel, padding=self.kernel_size // 2)
        return smooth_mask ** 0.25
