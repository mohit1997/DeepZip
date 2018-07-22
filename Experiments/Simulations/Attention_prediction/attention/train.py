import torch
from torch.autograd import Variable
 
def train(attention_model,train_loader,criterion,optimizer,epochs = 5,use_regularization = False,C=0,clip=False):
    """
        Training code
 
        Args:
            attention_model : {object} model
            train_loader    : {DataLoader} training data loaded into a dataloader
            optimizer       :  optimizer
            criterion       :  loss function. Must be BCELoss for binary_classification and NLLLoss for multiclass
            epochs          : {int} number of epochs
            use_regularizer : {bool} use penalization or not
            C               : {int} penalization coeff
            clip            : {bool} use gradient clipping or not
       
        Returns:
            accuracy and losses of the model
 
      
        """
    attention_model.train()
    losses = []
    accuracy = []
    for i in range(epochs):
        # print("Running EPOCH",i+1)
        total_loss = 0
        nllloss = 0
        n_batches = 0
        correct = 0
       
        for batch_idx,train in enumerate(train_loader):
 
            attention_model.hidden_state = attention_model.init_hidden()
            x,y = Variable(train[0]),Variable(train[1])
            x = x.cuda()
            y = y.cuda()
            y_pred,att = attention_model(x)
           
            #penalization AAT - I
            if use_regularization:
                attT = att.transpose(1,2)
                identity = torch.eye(att.size(1)).cuda()
                identity = Variable(identity.unsqueeze(0).expand(train_loader.batch_size,att.size(1),att.size(1)))
                penal = attention_model.l2_matrix_norm(torch.bmm(att, attT) - identity)
           
            
            if not bool(attention_model.type) :
                #binary classification
                #Adding a very small value to prevent BCELoss from outputting NaN's
                correct+=torch.eq(torch.round(y_pred.type(torch.DoubleTensor).cuda().squeeze(1)),y.type(torch.DoubleTensor).cuda()).sum().item()
                if use_regularization:
                    try:
                        loss1 = criterion(y_pred.type(torch.DoubleTensor).cuda().squeeze(1)+1e-8,y) 
                        loss2 = C * penal/train_loader.batch_size
                        loss = loss1 + loss2
                       
                    except RuntimeError:
                        raise Exception("BCELoss gets nan values on regularization. Either remove regularization or add very small values")
                else:
                    loss = criterion(y_pred.type(torch.DoubleTensor).cuda().squeeze(1),y)
                
            
            else:
                
                correct+=torch.eq(torch.max(y_pred.type(torch.LongTensor).cuda(),1)[1],y.type(torch.LongTensor).cuda()).sum().item()
                if use_regularization:
                    loss1 = criterion(y_pred,y.type(torch.LongTensor).cuda())
                    loss2 = (C * penal/train_loader.batch_size).type(torch.FloatTensor).cuda()
                    loss = loss1 + loss2
                else:
                    loss = criterion(y_pred,y.type(torch.LongTensor).cuda())
               
 
            total_loss+=loss.item()
            nllloss += loss1.item()
            optimizer.zero_grad()
            loss.backward()
           
            #gradient clipping
            if clip:
                torch.nn.utils.clip_grad_norm_(attention_model.parameters(),0.5)
            optimizer.step()
            n_batches+=1
            # print(nllloss/n_batches)
        torch.save(attention_model.state_dict(), "model_param.pkl")
        print("size", y_pred.size(), y.size())
        print("avg_loss is",total_loss/n_batches)
        print("nll_loss is",nllloss/n_batches)
        print("Accuracy of the model",correct/float(n_batches*train_loader.batch_size))
        losses.append(total_loss/n_batches)
        accuracy.append(correct/(n_batches*train_loader.batch_size))
    return losses,accuracy
 
 
def evaluate(attention_model,x_test,y_test):
    """
        cv results
 
        Args:
            attention_model : {object} model
            x_test          : {nplist} x_test
            y_test          : {nplist} y_test
       
        Returns:
            cv-accuracy
 
      
    """
   
    attention_model.batch_size = x_test.shape[0]
    attention_model.hidden_state = attention_model.init_hidden()
    x_test_var = Variable(torch.from_numpy(x_test).type(torch.LongTensor))
    y_test_pred,_ = attention_model(x_test_var)
    if bool(attention_model.type):
        y_preds = torch.max(y_test_pred,1)[1]
        y_test_var = Variable(torch.from_numpy(y_test).type(torch.LongTensor))
       
    else:
        y_preds = torch.round(y_test_pred.type(torch.DoubleTensor).squeeze(1))
        y_test_var = Variable(torch.from_numpy(y_test).type(torch.DoubleTensor))
       
    return torch.eq(y_preds,y_test_var).data.sum()/x_test_var.size(0)

def validate(attention_model, val_loader, criterion, use_regularization=False, C=0):
    attention_model.eval()
    losses = []
    accuracy = []
    total_loss = 0
    n_batches = 0
    correct = 0
    for batch_idx,train in enumerate(val_loader):
        attention_model.hidden_state = attention_model.init_hidden()
        x,y = Variable(train[0]),Variable(train[1])
        x = x.cuda()
        y = y.cuda()
        # print(x.size())

        y_pred,att = attention_model(x)
        # print(y_pred.size())

        if use_regularization:
            attT = att.transpose(1,2)
            identity = torch.eye(att.size(1)).cuda()
            identity = Variable(identity.unsqueeze(0).expand(val_loader.batch_size,att.size(1),att.size(1)))
            penal = attention_model.l2_matrix_norm(torch.bmm(att, attT) - identity)

        if attention_model.type == 0:
            correct+=torch.eq(torch.round(y_pred.type(torch.DoubleTensor).cuda().squeeze(1)),y.type(torch.DoubleTensor).cuda()).sum().item()
            loss = criterion(y_pred,y.type(torch.FloatTensor).cuda())
        else:    
            correct+=torch.eq(torch.max(y_pred.type(torch.LongTensor).cuda(),1)[1],y.type(torch.LongTensor).cuda()).sum().item()
            loss = criterion(y_pred,y.type(torch.LongTensor).cuda())
        n_batches += 1
        # print(n_batches)
        total_loss += loss.item()
    print(attention_model.type, n_batches)
    print("Validation avg_loss is",total_loss/n_batches)
    print("Validation Accuracy of the model",correct/float(n_batches*val_loader.batch_size))




def get_activation_wts(attention_model,x):
    """
        Get r attention heads
 
        Args:
            attention_model : {object} model
            x               : {torch.Variable} input whose weights we want
       
        Returns:
            r different attention weights
 
      
    """
    attention_model.batch_size = x.size(0)
    attention_model.hidden_state = attention_model.init_hidden()
    _,wts = attention_model(x)
    return wts