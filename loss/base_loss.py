import torch.nn as nn

from . import LOSS


@LOSS.register_module
class CosLoss(nn.Module):
    def __init__(self, avg=True, flat=True, lam=1):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity()
        self.lam = lam
        self.avg = avg
        self.flat = flat

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for teacher, student in zip(input1, input2):
            if self.flat:
                teacher = teacher.contiguous().view(teacher.shape[0], -1)
                student = student.contiguous().view(student.shape[0], -1)
            loss += (1 - self.cos_sim(teacher, student)).mean() * self.lam
        return loss / len(input1) if self.avg else loss
