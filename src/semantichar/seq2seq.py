import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import semantichar.data 
from semantichar import imagebind_model
from semantichar.imagebind_model import ModalityType

class Encoder(nn.Module):

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 seq_len: int):
        super().__init__()

        self.layer1 = nn.Conv1d(in_channels=d_input, out_channels=d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.act1 = nn.ReLU()

        self.layer2 = nn.Conv1d(in_channels=d_model, out_channels=d_output, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b,t,c = x.size()

        out = self.layer1(x.permute(0,2,1))
        out = self.act1(self.bn1(out))

        out = self.layer2(out)
        out = self.act2(self.bn2(out)) # (b, d_output, seq_len)

        return out.permute(0,2,1) # (b, seq_len, d_output)

class Decoder(nn.Module):
    
    def __init__(self, embed_dim, decoder_dim, vocab, encoder_dim, device, dropout=0.5):
      
        super(Decoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)  
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim, decoder_dim, bias=True)  
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.fc = nn.Linear(decoder_dim, self.vocab_size) 
        self.load_pretrained_embeddings()

    def load_pretrained_embeddings(self):

        inputs = {
            ModalityType.TEXT: semantichar.data.load_and_transform_text(self.vocab, self.device)
        }
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            embeddings = model(inputs)['text']
        self.embedding.weight = nn.Parameter(embeddings)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        
        mean_encoder_out = encoder_out.mean(dim=1) 
        h = self.init_h(mean_encoder_out) 
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  
        seq_len = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions.long()) 

        h, c = self.init_hidden_state(encoder_out)  

        decode_lengths = (caption_lengths - 1).tolist()
        
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths]) 
            h, c = self.decode_step(embeddings[:batch_size_t, t, :], \
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))  
            predictions[:batch_size_t, t, :] = preds
        return predictions, encoded_captions, decode_lengths, sort_ind