import torch
import numpy as np

def onehot(l, k, onehot_dict, word_list):
    encoding = np.zeros(k)
    for i, label in enumerate(onehot_dict[l]):
        encoding[i] = word_list.index(label)
    return encoding, len(onehot_dict[l])

def prepare_dataset(
    tr_data, 
    tr_label, 
    test_data, 
    test_label, 
    label_dictionary, 
):
    """
    Prepare training and testing datasets.
    Args:
        tr_data, test_data (numpy array): training and test data. size = (N, seq_len, channel_dim).
        tr_label, test_label (numpy array): training and test label. size = (N, ).
        label_dictionary (dict): mapping from label id to label token sequence, e.g., {"0": ["open", "door"],
                                "1": ["close", "fridge"]}.
    Returns:
        seq_len, dim, class_num, vocab_size, break_step (int): dataset-dependent parameters that describe
        time-series sequence length, channel dimension, number of classes, label name vocabulary size,
        longest label sequence length (i.e., maximum decoding step).
        word_list (list): list of tokens in the label sequences.
        pred_dict (dict): mapping from label token-id sequence to label id.
        seqs (torch.tensor): label token-id sequence for all classes.
        tr_data, test_data (torch.tensor): training and test data.
        tr_label, test_label (torch.tensor): training and test label.
        tr_text, test_text (torch.tensor): training and test token-id sequence.
        len_text, test_len_text (torch.tensor): training and test token-id sequence length. 
    """

    seq_len, dim, class_num = tr_data.shape[1], tr_data.shape[2], len(label_dictionary)
    break_step = 3
    word_set = set()
    for v_list in label_dictionary.values():
        break_step = max(len(v_list) + 2, break_step)
        for v in v_list:
            word_set.add(v)
    word_list = ['start'] + list(word_set) + ['end']
    pred_dict = {"#".join(map(str, [word_list.index(i) for i in v])): k for k, v in label_dictionary.items()}
    vocab_size = len(word_list)

    onehot_dict = {k:['start']+v+['end'] for k, v in label_dictionary.items()}
    tr_text = np.stack([onehot(l, break_step, onehot_dict, word_list)[0] for l in tr_label], axis=0)
    len_text = np.vstack([onehot(l, break_step, onehot_dict, word_list)[1] for l in tr_label])
    test_text = np.stack([onehot(l, break_step, onehot_dict, word_list)[0] for l in test_label], axis=0)
    test_len_text = np.vstack([onehot(l, break_step, onehot_dict, word_list)[1] for l in test_label])

    tr_data, test_data, tr_label, test_label, tr_text, test_text, len_text, test_len_text = \
        torch.from_numpy(tr_data).float(), torch.from_numpy(test_data).float(), \
        torch.from_numpy(tr_label).float(), torch.from_numpy(test_label).float(), \
        torch.from_numpy(tr_text).float(), torch.from_numpy(test_text).float(), \
        torch.from_numpy(len_text), torch.from_numpy(test_len_text)

    print('load dataset')
    print(tr_data.shape, test_data.shape)

    seqs = torch.zeros((class_num, break_step)).long()
    for i, v in enumerate(onehot_dict.values()):
        for j in range(len(v)):
            seqs[i,j] = word_list.index(v[j])
    
    return seq_len, dim, class_num, vocab_size, break_step, word_list, pred_dict, seqs, \
            tr_data, test_data, tr_label, test_label, \
            tr_text, test_text, len_text, test_len_text
