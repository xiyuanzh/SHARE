import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence

from semantichar.utils import all_label_augmentation

def DataBatch(data, label, text, l, batchsize, shuffle=True):
    
    n = data.shape[0]
    if shuffle:
        index = np.random.permutation(n)
    else:
        index = np.arange(n)
    for i in range(int(np.ceil(n/batchsize))):
        inds = index[i*batchsize : min(n,(i+1)*batchsize)]
        yield data[inds], label[inds], text[inds], l[inds]
        
def trainer(opt, 
            enc, 
            dec, 
            cross_entropy, 
            optimizer, 
            tr_data, 
            tr_label, 
            tr_text, 
            len_text, 
            break_step, 
            vocab_size, 
            device
):
    """
    Train the model.
    Args:
        opt: user-specified configurations.
        enc: encoder of the model.
        dec: decoder of the model.
        cross_entropy: loss function.
        optimizer: optimizer (default is Adam).
        tr_data, tr_label, tr_text, len_text: training data, label, label sequence, length of the label sequence. 
        break_step: length of the longest label sequence length (i.e., maximum decoding step).
        vocab_size: label name vocabulary size.
        device: cuda or cpu.
    """

    enc.train()
    dec.train()  

    total_loss = 0
    for batch_data, batch_label, batch_text, batch_len in \
        DataBatch(tr_data, tr_label, tr_text, len_text, opt.batchSize):
        
        batch_text = all_label_augmentation(batch_text, opt.prob, break_step, vocab_size)

        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)
        batch_text = batch_text.to(device)
        batch_len = batch_len.to(device)

        enc_hidden = enc(batch_data)
        pred, batch_text_sorted, decode_lengths, sort_ind \
            = dec(enc_hidden, batch_text, batch_len)
        
        targets = batch_text_sorted[:, 1:]

        pred, *_ = pack_padded_sequence(pred, decode_lengths, batch_first=True)
        targets, *_ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = cross_entropy(pred, targets.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += len(batch_data) * loss.item()

    total_loss /= len(tr_data)
    
    return total_loss 
  

def evaluate(opt, 
             enc, 
             dec, 
             test_data, 
             test_label, 
             test_text, 
             test_len_text, 
             pred_dict, 
             seqs, 
             break_step, 
             class_num, 
             vocab_size, 
             device,
             load=True,
):
    
    """
    Evaluate the model.
    Args:
        opt: user-specified configurations.
        enc: encoder of the model.
        dec: decoder of the model.
        test_data, test_label, test_text, test_len_text: test data, label, label sequence, length of the label sequence.
        pred_dict: mapping from label token-id sequence to label id.
        seqs: label token-id sequence for all classes.
        break_step: length of the longest label sequence length (i.e., maximum decoding step).
        class_num: number of classes.
        vocab_size: label name vocabulary size.
        device: cuda or cpu.
        load: load saved model weights or not.
    """

    enc.eval()
    dec.eval()  

    if load:
        enc.load_state_dict(torch.load(opt.model_path + opt.run_tag + 'enc')) 
        dec.load_state_dict(torch.load(opt.model_path + opt.run_tag + 'dec'))  

    hypotheses = list()

    batch_size = test_data.size(0)

    pred_whole = torch.zeros_like(test_label)

    seqs = seqs.to(device)

    for batch_idx, (batch_data, batch_label, batch_text, batch_len) in \
        enumerate(DataBatch(test_data, test_label, test_text, test_len_text, opt.batchSize, shuffle=False)):

        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)
        batch_text = batch_text.to(device)
        batch_len = batch_len.to(device)
        
        batch_size = batch_data.size(0)
        encoder_out = enc(batch_data)  # (batch_size, enc_seq_len, encoder_dim)
        enc_seq_len = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        
        encoder_out = encoder_out.unsqueeze(1).expand(batch_size, class_num, enc_seq_len, encoder_dim) 
        encoder_out = encoder_out.reshape(batch_size*class_num, enc_seq_len, encoder_dim) 

        k_prev_words = seqs[:, 0].unsqueeze(0).expand(batch_size, class_num).long()  # (batch_size, class_num)
        k_prev_words = k_prev_words.reshape(batch_size*class_num, 1) # (batch_size*class_num, 1)
        
        h, c = dec.init_hidden_state(encoder_out) 
        
        seq_scores = torch.zeros((batch_size, class_num)).to(device)
        
        for step in range(1, break_step):
            embeddings = dec.embedding(k_prev_words).squeeze(1)  # (batch_size*class_num, embed_dim)
            
            h, c = dec.decode_step(embeddings, (h, c))
        
            scores = dec.fc(h.reshape(batch_size, class_num, -1))  # (batch_size, class_num, vocab_size)
            scores = F.log_softmax(scores, dim=-1)
        
            k_prev_words = seqs[:, step].unsqueeze(0).expand(batch_size, class_num).long()
            for batch_i in range(batch_size):
                for class_i in range(class_num):
                    if k_prev_words[batch_i, class_i] != 0:
                        seq_scores[batch_i, class_i] += scores[batch_i, class_i, k_prev_words[batch_i, class_i]]
            k_prev_words = k_prev_words.reshape(batch_size*class_num, 1) # (batch_size*class_num, 1)
        
        max_indices = seq_scores.argmax(dim=1)
        for batch_i in range(batch_size):
            max_i = max_indices[batch_i]
            seq = seqs[max_i].tolist()
            
            # Hypotheses and Predictions
            hypotheses.append([w for w in seq if w not in {0, vocab_size - 1}])
            pred_whole[batch_i + batch_idx * opt.batchSize] = pred_dict["#".join(map(str, hypotheses[-1]))]

    acc = accuracy_score(test_label.cpu().numpy(), pred_whole.cpu().numpy())
    prec = precision_score(test_label.cpu().numpy(), pred_whole.cpu().numpy(), average='macro')
    rec = recall_score(test_label.cpu().numpy(), pred_whole.cpu().numpy(), average='macro')
    f1 = f1_score(test_label.cpu().numpy(), pred_whole.cpu().numpy(), average='macro')
    
    return acc, prec, rec, f1
