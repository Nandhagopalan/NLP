import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from dataset import NerDataset, pad, HParams
import os
import numpy as np
from transformers import BertConfig
import config
from collections import OrderedDict


# prepare biobert dict 
# tmp_d = torch.load(config.BERT_WEIGHTS, map_location='cpu')
# state_dict = OrderedDict()
# for i in list(tmp_d.keys())[:199]:
#     x = i
#     if i.find('bert') > -1:
#         x = '.'.join(i.split('.')[1:])
#     state_dict[x] = tmp_d[i]
    

def train(model, iterator, optimizer, criterion,device):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        
        _y = y # for monitoring
        y = y.to(device)

        optimizer.zero_grad()
        logits, yhat = model(x) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i==0:
            print("=====sanity check======")
            print("x:", x.cpu().numpy()[0])
            print("words:", words[0])
            print("tokens:", hp.tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0]))
            print("y:", _y.cpu().numpy()[0])
            print("is_heads:", is_heads[0])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])


        if i%10==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")

def eval(model, iterator, f,device):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            _,y_hat = model(x)  # y_hat: (N, T)


            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open(f, 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [hp.idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true =  np.array([hp.tag2idx[line.split()[1]] for line in open(f, 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([hp.tag2idx[line.split()[2]] for line in open(f, 'r').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    final = f + ".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
    with open(final, 'w') as fout:
        result = open(f, "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")

    os.remove(f)

    print("precision=%.2f"%precision)
    print("recall=%.2f"%recall)
    print("f1=%.2f"%f1)
    return precision, recall, f1

if __name__=="__main__":
    
    
    train_dataset = NerDataset("../input/train.tsv", 'bc5cdr')  # here bc5cdr is dataset type
    eval_dataset = NerDataset("../input/test.tsv", 'bc5cdr')
    hp = HParams('bc5cdr')

    # Define model 
    #config = BertConfig(vocab_size_or_config_json_file=config.BERT_CONFIG_FILE)
    
    model = Net(config = config.BERT_CONFIG_FILE, weight =config.BERT_WEIGHTS , vocab_len = len(hp.VOCAB), device=hp.device)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    # update with already pretrained weight


    
    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, hp.n_epochs+1):
        train(model, train_iter, optimizer, criterion,hp.device)
        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
        fname = os.path.join('checkpoints', str(epoch))
        precision, recall, f1 = eval(model, eval_iter, fname,hp.device)
        torch.save(model.state_dict(), f"{fname}.pt")