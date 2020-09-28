import math
import torch
from torch.utils.data import Dataset
from .nn_modules import *

class dataloader(Dataset):
    '''
    dataloader
    '''
    
    def __init__(self, X, X_length, y1, y2, y3):
        '''
        :param output_folder: 
        :param split: 'train', 'dev', or 'test'
        '''
#         self.split = split
#         assert self.split in {'train', 'dev', 'test'}
        
        self.dataset = X
        self.length = X_length
        self.label_1 = y1
        self.label_2 = y2
        self.label_3 = y3

        self.dataset_size = self.dataset.shape[0]
        
    def __getitem__(self, i):

        sentence = self.dataset[i] # sentence shape [max_len]
        sentence_length = self.length[i]
        sentence_label_1 = self.label_1[i]
        sentence_label_2 = self.label_2[i]
        sentence_label_3 = self.label_3[i]
        
        return sentence, sentence_length, sentence_label_1, sentence_label_2, sentence_label_3

    def __len__(self):
        
        return self.dataset_size
    
def adjust_learning_rate(optimizer, current_epoch):
    '''
    learning rate decay
    '''
    frac = float(current_epoch - 20) / 50
    shrink_factor = math.pow(0.5, frac)
    
    print("DECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor

    print("The new learning rate is {}".format(optimizer.param_groups[0]['lr']))
    
def accuracy(logits, targets):
    '''
    :param logits: (batch_size, class_num)
    :param targets: (batch_size, class_num)
    :return: 
    '''
    pred = logits.data > 0.5
    true = targets.data > 0.5
    return (true == pred).sum().item()/(pred.shape[1] * pred.shape[0])

class AverageMeter():
    '''
    batch average acc
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0. #value
        self.avg = 0. #average
        self.sum = 0. #sum
        self.count = 0 #count

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count 

def clip_gradient(optimizer, grad_clip):
    """
    Gradient clip
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                # inplace, no new tensors created
                # gradient cliped to (-grad_clip, grad_clip)
                param.grad.data.clamp_(-grad_clip, grad_clip)
                
def train(train_loader, model, criterion, optimizer, epoch, print_freq, device, grad_clip=None):
    '''
    Train one epoch
    '''
    # enable dropout
    model.train()
    
    losses = AverageMeter()  # average loss for a batch
    accs = AverageMeter()  # average acc for a batch
    
    for i, (seqs, seqs_len, labels_1, labels_2, labels_3) in enumerate(train_loader):
        index = torch.flip(np.argsort(seqs_len), dims = [0])
        seqs = seqs[index]
        seqs_len = seqs_len[index]
        labels_1 = labels_1[index]
        labels_2 = labels_2[index]
        labels_3 = labels_3[index]
        # move to CPU/GPU
        seqs = seqs.to(device)
        seqs_len = seqs_len.to(device)
        
        labels_1 = labels_1.to(device)
        labels_2 = labels_2.to(device)
        labels_3 = labels_3.to(device)
        
        # forward
        logits_1, logits_2, logits_3 = model(seqs, seqs_len)
            
        logits_2 = logits_2[labels_2.sum(dim = 1) >= 1]
        logits_3 = logits_3[labels_3.sum(dim = 1) >= 1]
        labels_2 = labels_2[labels_2.sum(dim = 1) >= 1]
        labels_3 = labels_3[labels_3.sum(dim = 1) >= 1]
        
        # loss
        loss_1 = criterion(logits_1, labels_1)
        acc1 = accuracy(logits_1, labels_1)
        if labels_2.size()[0] == 0:
            loss_2 = 0
            acc2 = 0
        else:
            loss_2 = criterion(logits_2, labels_2)
            acc2 = accuracy(logits_2, labels_2)
        if labels_3.size()[0] == 0:   
            loss_3 = 0
            acc3 = 0
        else:
            loss_3 = criterion(logits_3, labels_3)
            acc3 = accuracy(logits_3, labels_3)
        loss =  (loss_1 + loss_2 + loss_3)/3
        acc = (acc1 + acc2 + acc3)/3
        # backprop
        optimizer.zero_grad()
        loss.backward()

        # grad_clip
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        
        # update optimizer
        optimizer.step()
        
        # update performance
        accs.update(acc)
        losses.update(loss.item())
        
        # print
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          loss=losses,
                                                                          acc=accs))
            
def validate(val_loader, model, criterion, print_freq, device):
    '''
    validate on one epoch
    '''
    
    #disenble drop off
    model = model.eval()

    losses = AverageMeter()  
    accs = AverageMeter()  

    # no gradient calculation
    with torch.no_grad():
        for i, (seqs, seqs_len, labels_1, labels_2, labels_3) in enumerate(val_loader):
            index = torch.flip(np.argsort(seqs_len), dims = [0])
            seqs = seqs[index]
            seqs_len = seqs_len[index]
            labels_1 = labels_1[index]
            labels_2 = labels_2[index]
            labels_3 = labels_3[index]

            # move to CPU/GPU
            seqs = seqs.to(device)
            seqs_len = seqs_len.to(device)

            labels_1 = labels_1.to(device)
            labels_2 = labels_2.to(device)
            labels_3 = labels_3.to(device)

            # forward
            logits_1, logits_2, logits_3 = model(seqs, seqs_len)

            logits_2 = logits_2[labels_2.sum(dim = 1) >= 1]
            logits_3 = logits_3[labels_3.sum(dim = 1) >= 1]
            labels_2 = labels_2[labels_2.sum(dim = 1) >= 1]
            labels_3 = labels_3[labels_3.sum(dim = 1) >= 1]
            # loss
            loss_1 = criterion(logits_1, labels_1)
            acc1 = accuracy(logits_1, labels_1)
            if labels_2.size()[0] == 0:
                loss_2 = 0
                acc2 = 0
            else:
                loss_2 = criterion(logits_2, labels_2)
                acc2 = accuracy(logits_2, labels_2)
            if labels_3.size()[0] == 0:   
                loss_3 = 0
                acc3 = 0
            else:
                loss_3 = criterion(logits_3, labels_3)
                acc3 = accuracy(logits_3, labels_3)
            
            loss =  (loss_1 + loss_2 + loss_3)/3
            losses.update(loss.item())

            acc = (acc1 + acc2 + acc3)/3
            accs.update(acc)
            

            if i % print_freq  == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(i, len(val_loader),
                                                                                loss=losses, acc=accs))
        # overall acc
        print('LOSS - {loss.avg:.3f}, ACCURACY - {acc.avg:.3f}\n'.format(loss=losses, acc=accs))

    return accs.avg

def train_eval(opt, train_target_data, valid_target_data, hierarchical=True, functions=None, go=None):
    '''
    Train with validation
    '''
    # best accuracy
    best_acc = 0.

    # epoch
    start_epoch = 0
    epochs = opt.epochs
    epochs_since_improvement = 0  
    
    # initialze model

    X, X_length, y1, y2, y3 = train_target_data
    X_val, X_val_length, y_val_1, y_val_2, y_val_3 = valid_target_data
    
    model = CrossStitchModel(in_channels=X.shape[2], out_channels=opt.ResNet_out_channels, kernel_size=opt.ResNet_kernel_size, 
                  n_layers=opt.ResNet_n_layers, n_class=[y1.shape[1], y2.shape[1], y3.shape[1]], n_hidden_state=opt.LSTM_n_hidden_state, 
                  use_gru=True, lstm_dropout=0, n_lstm_layers=1, activation='sigmoid', hierarchical=True, functions=functions, go=go)
    
    model = model.to(torch.double)

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    
    # move to CPU/GPU
    model = model.to(opt.device)
    
    # loss function
    criterion = nn.BCELoss().to(opt.device)
    
    train_data = dataloader(X, X_length, y1, y2, y3)
    val_data = dataloader(X_val, X_val_length, y_val_1, y_val_2, y_val_3)
    
    # Train/valide data

    train_loader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=opt.batch_size, 
                        shuffle=True,
                        num_workers = 1,
                        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
                        val_data,
                        batch_size=opt.batch_size, 
                        shuffle=True,
                        num_workers = 1,
                        pin_memory=True)
    
    # Epochs
    for epoch in range(start_epoch, epochs):
        
        # decay learning rate
        if epoch > opt.decay_epoch:
            adjust_learning_rate(optimizer, epoch)
        
        # early stopping
        if epochs_since_improvement == opt.improvement_epoch:
            break
        
        # train on one epoch
        train(train_loader=train_loader, model=model, criterion=criterion, 
              optimizer=optimizer, epoch=epoch, print_freq=opt.print_freq, 
              device=opt.device, grad_clip=opt.grad_clip)

        # validate on one epoch
        recent_acc = validate(val_loader=val_loader, model=model, criterion=criterion, 
                              print_freq=opt.print_freq, device=opt.device)
        
        # check improvements
        is_best = recent_acc > best_acc
        best_acc = max(recent_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        
        torch.save(model.state_dict(), '{}/trained_model_epoch:{}_perf:{}.pkl'.format(opt.save_model_path, epoch, recent_acc))