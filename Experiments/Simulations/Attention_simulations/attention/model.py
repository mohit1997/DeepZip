import torch,keras
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data_utils
 
class StructuredSelfAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """
   
    def __init__(self,batch_size,lstm_hid_dim,d_a,r,max_len,emb_dim=4,vocab_size=None,use_pretrained_embeddings = False,embeddings=None,type=0,n_classes = 1):
        """
        Initializes parameters suggested in paper
 
        Args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            max_len     : {int} number of lstm timesteps
            emb_dim     : {int} embeddings dimension
            vocab_size  : {int} size of the vocabulary
            use_pretrained_embeddings: {bool} use or train your own embeddings
            embeddings  : {torch.FloatTensor} loaded pretrained embeddings
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes
 
        Returns:
            self
 
        Raises:
            Exception
        """
        super(StructuredSelfAttention,self).__init__()
        self.lstm_layers = 3
        self.embeddings,emb_dim = self._load_embeddings(use_pretrained_embeddings,embeddings,vocab_size,emb_dim)
        self.lstm = torch.nn.LSTM(emb_dim,lstm_hid_dim,self.lstm_layers, dropout=0.1, batch_first=True)
        self.linear_first = torch.nn.Linear(lstm_hid_dim,d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(lstm_hid_dim,self.n_classes)
        self.batch_size = batch_size       
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.hidden_state = self.init_hidden()
        self.r = r
        self.type = type
        self.bn1 = torch.nn.LayerNorm([max_len, lstm_hid_dim])
        self.bn2 = torch.nn.LayerNorm([max_len, d_a])
        self.bn3 = torch.nn.LayerNorm([max_len, r])
        self.conv1 = torch.nn.Conv1d(r, r, 3, padding=1)
        self.convinp = torch.nn.Conv1d(emb_dim, 1, 3, padding=1)
        self.drop1 = torch.nn.Dropout2d(p=0.2)
        # self.inplinear = 
        self.inplinear = torch.nn.Sequential(
            torch.nn.Linear(lstm_hid_dim+max_len, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, self.n_classes),
          )
        # self.bn2 = torch.nn.LayerNorm()
                 
    def _load_embeddings(self,use_pretrained_embeddings,embeddings,vocab_size,emb_dim):
        """Load the embeddings based on flag"""
       
        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")
           
        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")
   
        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size,emb_dim,padding_idx=0)
            
        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(embeddings)
            emb_dim = embeddings.size(1)
            
        return word_embeddings,emb_dim
       
        
    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
 
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied
 
        Returns:
            softmaxed tensors
 
       
        """
 
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
       
        
    def init_hidden(self):
        return (Variable(torch.zeros(self.lstm_layers,self.batch_size,self.lstm_hid_dim)).cuda(),Variable(torch.zeros(self.lstm_layers,self.batch_size,self.lstm_hid_dim)).cuda())
       
        
    def forward(self,x):
        embeddings = self.embeddings(x)
        inp = x.type(torch.FloatTensor).cuda()
        # conv_embeddings = self.convinp(embeddings.transpose(1, 2)).transpose(1, 2)
        outputs, self.hidden_state = self.lstm(embeddings.view(self.batch_size,self.max_len,-1),self.hidden_state)
        



        x = self.bn2(torch.tanh((self.linear_first(self.bn1(outputs)))))
        x = self.bn3((self.linear_second(x)))
        # print(x.size())  
        x = self.softmax(x,1)    
        attention = x.transpose(1,2)
        sentence_embeddings = torch.bmm(attention, outputs)
        # print(sentence_embeddings.size())
        sentence_embeddings = self.conv1(sentence_embeddings)
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r
        # fin_embeddings = torch.cat(sente)
        # print(sentence_embeddings.size(), avg_sentence_embeddings.size(), attention.size(), embeddings.size())
        # mean_attention = embeddings*torch.mean(attention.transpose(1, 2), dim=2, keepdim=True)
        mean_attention = self.convinp(embeddings.transpose(1, 2)).transpose(1, 2)
        mean_attention = mean_attention.view(self.batch_size, -1)

        out = torch.cat((mean_attention, avg_sentence_embeddings), dim=1)

        # print(sentence_embeddings.size(), avg_sentence_embeddings.size(), attention.size(), mean_attention.size())

       
        if not bool(self.type):
            output = F.sigmoid(self.inplinear(out))
           
            return output,attention
        else:
            # return F.log_softmax(self.linear_final(avg_sentence_embeddings), dim=1), attention
            return F.log_softmax(self.inplinear(out), dim=1), attention
       
	   
	#Regularization
    def l2_matrix_norm(self,m):
        """
        Frobenius norm calculation
 
        Args:
           m: {Variable} ||AAT - I||
 
        Returns:
            regularized value
 
       
        """
        return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5).type(torch.DoubleTensor).cuda()