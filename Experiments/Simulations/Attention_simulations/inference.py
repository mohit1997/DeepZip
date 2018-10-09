#You can write your own classification file to use the module
from attention.model import StructuredSelfAttention
from attention.train import train,get_activation_wts,evaluate,validate, predict
from utils.pretrained_glove_embeddings import load_glove_embeddings
from utils.data_loader import load_data_set
from visualization.attention_visualization import createHTML
import torch
import numpy as np
from torch.autograd import Variable
from keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
import torch.utils.data as data_utils
import os,sys
import json
 
classified = False
classification_type = sys.argv[1]
 
def json_to_dict(json_set):
    for k,v in json_set.items():
        if v == 'False':
            json_set[k] = False
        elif v == 'True':
            json_set[k] = True
        else:
            json_set[k] = v
    return json_set
 
 
with open('config.json', 'r') as f:
    params_set = json.load(f)
 
with open('model_params.json', 'r') as f:
    model_params = json.load(f)
 
params_set = json_to_dict(params_set)
model_params = json_to_dict(model_params)
 
print("Using settings:",params_set)
print("Using model settings",model_params)
 
def visualize_attention(wts,x_test_pad,word_to_id,filename):
    wts_add = torch.sum(wts,1)
    wts_add_np = wts_add.cpu().data.numpy()
    wts_add_list = wts_add_np.tolist()
    id_to_word = {v:k for k,v in word_to_id.items()}
    text= []
    for test in x_test_pad:
        text.append(" ".join([id_to_word.get(i) for i in test]))
    createHTML(text, wts_add_list, filename)
    print("Attention visualization created for {} samples".format(len(x_test_pad)))
    return
 
def binary_classfication(attention_model,train_loader,val_loader=None, epochs=5,use_regularization=True,C=1.0,clip=True):
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(attention_model.parameters())
    for i in range(10):
        train(attention_model,train_loader,loss,optimizer,1,use_regularization,C,clip)
        if val_loader is not None:
            validate(attention_model, val_loader, loss, use_regularization, C=1.0)
 
def multiclass_classification(attention_model,train_loader, val_loader=None, epochs=5,use_regularization=True,C=1.0,clip=True):
    loss = torch.nn.NLLLoss()
    optimizer = torch.optim.RMSprop(attention_model.parameters())
    for i in range(1):
        print("--------------------: Validation Results : ----------------")
        # train(attention_model,train_loader,loss,optimizer,1,use_regularization,C,clip)
        if val_loader is not None:
            out = predict(attention_model, val_loader, loss, use_regularization, C=1.0)

    # res = [i.detach().cpu().numpy() for i in out]
    res = np.concatenate(out, axis=0)
    a = 0.2*np.ones((30, 5)).astype(np.float32)
    print(a.dtype)
    res = np.concatenate([a, res], axis=0)
    print(res.dtype)
    print(res.shape)
    np.save('prob', res)

 

model_params['timesteps'] = 30
MAXLENGTH = model_params['timesteps']
if classification_type =='binary':
    model_params['batch_size'] = 64
    # print("batch size", )
    train_loader, val_loader, x_test_pad,y_test,word_to_id = load_data_set(0,MAXLENGTH,model_params["vocab_size"],model_params['batch_size']) #loading imdb dataset
 
 
    if params_set["use_embeddings"]:
        embeddings = load_glove_embeddings("glove/glove.6B.50d.txt",word_to_id,50)
    else:
        embeddings = None
    #Can use pretrained embeddings by passing in the embeddings and setting the use_pretrained_embeddings=True
    attention_model = StructuredSelfAttention(batch_size=train_loader.batch_size,lstm_hid_dim=model_params['lstm_hidden_dimension'],d_a = model_params["d_a"],r=params_set["attention_hops"],emb_dim=250, vocab_size=len(word_to_id),max_len=MAXLENGTH,type=0,n_classes=1,use_pretrained_embeddings=params_set["use_embeddings"],embeddings=embeddings)
 
    #Can set use_regularization=True for penalization and clip=True for gradient clipping
    attention_model.cuda()
    binary_classfication(attention_model,train_loader=train_loader, val_loader=val_loader, epochs=params_set["epochs"],use_regularization=params_set["use_regularization"],C=params_set["C"],clip=params_set["clip"])
    classified = True
    #wts = get_activation_wts(binary_attention_model,Variable(torch.from_numpy(x_test_pad[:]).type(torch.LongTensor)))
    #print("Attention weights for the testing data in binary classification are:",wts)
 
 
if classification_type == 'multiclass':
    train_loader,train_set,test_set,x_test_pad,word_to_id = load_data_set(1,MAXLENGTH,model_params["vocab_size"],model_params['batch_size']) #load the reuters dataset
    #Using pretrained embeddings
    if params_set["use_embeddings"]:
        embeddings = load_glove_embeddings("glove/glove.6B.50d.txt",word_to_id,50)
    else:
        embeddings = None
    attention_model = StructuredSelfAttention(batch_size=train_loader.batch_size,lstm_hid_dim=model_params['lstm_hidden_dimension'],d_a = model_params["d_a"],r=params_set["attention_hops"],vocab_size=len(word_to_id),max_len=MAXLENGTH,type=1,n_classes=46,use_pretrained_embeddings=params_set["use_embeddings"],embeddings=embeddings)
 
    #Using regularization and gradient clipping at 0.5 (currently unparameterized)
    multiclass_classification(attention_model,train_loader,epochs=params_set["epochs"],use_regularization=params_set["use_regularization"],C=params_set["C"],clip=params_set["clip"])
    classified=True
    #wts = get_activation_wts(multiclass_attention_model,Variable(torch.from_numpy(x_test_pad[:]).type(torch.LongTensor)))
    #print("Attention weights for the data in multiclass classification are:",wts)


if classification_type == 'dna':
    model_params['batch_size'] = 512
    model_params["vocab_size"] = 5
    n_classes = 5
    emb_dim = 10

    train_loader,val_loader,x_test_pad,word_to_id = load_data_set('dna',MAXLENGTH,model_params["vocab_size"], 10) #load the reuters dataset
    #Using pretrained embeddings
    print(len(word_to_id))
    params_set["attention_hops"] = 20
    params_set["use_embeddings"] = False
    model_params['lstm_hidden_dimension'] = 50
    model_params['d_a'] = 50
    params_set["C"] = 0.03
    if params_set["use_embeddings"]:
        embeddings = load_glove_embeddings("glove/glove.6B.50d.txt",word_to_id,50)
    else:
        embeddings = None
    attention_model = StructuredSelfAttention(batch_size=train_loader.batch_size,lstm_hid_dim=model_params['lstm_hidden_dimension'],d_a = model_params["d_a"],r=params_set["attention_hops"],emb_dim=emb_dim, vocab_size=len(word_to_id),max_len=MAXLENGTH,type=1,n_classes=n_classes,use_pretrained_embeddings=params_set["use_embeddings"],embeddings=embeddings)
 
    #Using regularization and gradient clipping at 0.5 (currently unparameterized)
    attention_model.load_state_dict(torch.load('model_param'))
    attention_model.cuda()
    attention_model.batch_size = 10
    multiclass_classification(attention_model,train_loader,val_loader,epochs=params_set["epochs"],use_regularization=params_set["use_regularization"],C=params_set["C"],clip=params_set["clip"])
    classified=False
    #wts = get_activation_wts(multiclass_attention_model,Variable(torch.from_numpy(x_test_pad[:]).type(torch.LongTensor)))
    #print("Attention weights for the data in multiclass classification are:",wts)


if classified:
    test_last_idx = 100
    print(x_test_pad)
    wts = get_activation_wts(attention_model,Variable(torch.from_numpy(x_test_pad[:test_last_idx]).type(torch.LongTensor).cuda()))
    print(wts.size())
    visualize_attention(wts,x_test_pad[:test_last_idx],word_to_id,filename='attention.html')