import torch
import torch.nn.functional as F
import numpy as np

def all_label_augmentation(batch_text, prob, break_step, vocab_size):

    """
    Label augmentation. 
    Args:
        batch_text (torch.tensor): batch text token-id sequence.
        prob (float): probability for replacing token-id sequence with single token-id.
        break_step (int): longest label sequence length (i.e., maximum decoding step).
        vocab_size (int): label name vocabulary size.
    Returns:
        aug_batch_text (torch.tensor): batch text token-id sequence after label augmentation. 
    """

    valid = np.arange(1, vocab_size - 1)
    prob = np.full((len(batch_text)), prob / 2)
    zeros = np.full((len(batch_text)), 0)[:,np.newaxis]
    end = np.full((len(batch_text)), vocab_size - 1)[:,np.newaxis]

    batch_text = batch_text.detach().cpu().numpy()
    idx = np.random.choice(np.arange(1,batch_text.shape[1] - 1),size=len(batch_text))
    aug_batch_text = np.array([batch_text[i, idx[i]] for i in range(len(batch_text))])
    new_zeros = np.concatenate([zeros for _ in range(break_step - 3)], axis=1)
    aug_batch_text_pad = np.concatenate((zeros, aug_batch_text[:,np.newaxis], end, new_zeros), axis=1) 
    rand = np.random.uniform(size=len(batch_text))
    aug_batch_text = np.where(np.logical_and(rand < prob, np.isin(aug_batch_text, valid))[:,np.newaxis], aug_batch_text_pad, batch_text)

    return torch.from_numpy(aug_batch_text)

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def count_memory(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    return mem, mem_params, mem_bufs