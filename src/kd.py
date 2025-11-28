"""Knowledge Distillation objective."""
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, y_true, alpha=0.6, temperature=2.0):
    """alpha * CE + (1-alpha) * T^2 * KL(softmax(t/T) || softmax(s/T))"""
    ce = F.cross_entropy(student_logits, y_true)
    T = temperature
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean")
    return alpha * ce + (1 - alpha) * (T * T) * kl
