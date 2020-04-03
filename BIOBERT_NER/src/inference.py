import torch
import config
from dataset import NerTestDataset, pad, HParams
from torch.utils import data
from model import Net
import pandas as pd

hp = HParams('bc5cdr')

eval_dataset = NerTestDataset("../input/output.tsv", 'bc5cdr')
eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=2,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)



# Define model 
#config = BertConfig(vocab_size_or_config_json_file=config.BERT_CONFIG_FILE)

model1 = Net(config = config.BERT_CONFIG_FILE, weight =config.BERT_WEIGHTS , vocab_len = len(hp.VOCAB), device=hp.device)

device = torch.device("cuda")
#model1 = model()
model1.to(device)
#model = nn.DataParallel(model)
model1.load_state_dict(torch.load(config.TRAINED_MODEL))


model1.eval()

Words, Is_heads, Tags, Y_hat = [], [], [], []
with torch.no_grad():
    for i, batch in enumerate(eval_iter):
        words, x, is_heads, tags, y, seqlens = batch

        _, y_hat = model1(x)  # y_hat: (N, T)

        Words.extend(words)
        Is_heads.extend(is_heads)
        Y_hat.extend(y_hat.cpu().numpy().tolist())

## gets results and save
#with open('../output/test.txt', 'w') as fout:
out_words=[]
tags=[]

for words, is_heads, y_hat in zip(Words, Is_heads, Y_hat):
    y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
    preds = [hp.idx2tag[hat] for hat in y_hat]
    out_words.extend(words.split(' '))
    tags.extend(preds)
    #print(preds)
    #print(words)


testdf=pd.DataFrame()
testdf['words']=out_words
testdf['tags']=tags

testdf.to_csv('../ouput/pred.csv',index=None)