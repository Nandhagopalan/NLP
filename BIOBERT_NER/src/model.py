import torch
import torch.nn as nn
from transformers import BertModel

class Net(nn.Module):
    def __init__(self, config, weight, vocab_len, device = 'cpu'):
        super().__init__()
        self.bert = BertModel.from_pretrained(weight, config=config)
        # if bert_state_dict is not None:
        #     self.bert.load_state_dict(bert_state_dict)
        #self.bert.eval()
        self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
        self.fc = nn.Linear(768, vocab_len)
        self.device = device

    def forward(self, x):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        

        with torch.no_grad():
            encoded_layers, _ = self.bert(x)
            #enc = encoded_layers[-1]
    
        enc, _ = self.rnn(encoded_layers)
        
        logits = self.fc(enc)

        y_hat = logits.argmax(-1)
        return logits, y_hat