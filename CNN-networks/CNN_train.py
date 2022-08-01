import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter   
import argparse
from torch.autograd import Variable
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import random
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import CNN_metrics as metrics
EPOCH = 10
pre_epoch = 0  
BATCH_SIZE = 32      
LR = 0.001  
image_size = 224
class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()
        w = torch.Tensor([0.5,0.2])
        self.paras = (nn.Parameter(w)) 
    def forward(self,x1,x2):
        weight = torch.sigmoid(self.paras)
        y = weight[0]*x1 + weight[1]*x2 
        return y

def get_att_dis(target, behaviored):
    attention_distribution = []
    for i in range(behaviored.size(0)):
        attention_score = torch.cosine_similarity(target, behaviored[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度
        attention_distribution.append(attention_score)
    attention_distribution = torch.Tensor(attention_distribution)
    return attention_distribution

def simi(z,label,class_nums):
    #input:batch*channel
    #label:batch*1
    batch_size = z.shape[0]
    sort = list(label.cpu().numpy().astype(int))
    y = {}
    for i in range(class_nums):
        y.setdefault(str(i),[])
    # y = {"0":[],"1":[],"2":[],"3":[]}
    for i in range(batch_size):
        y[str(sort[i])].append(i)
    class_inter = torch.Tensor([0]).cuda()
    class_outer = torch.Tensor([0]).cuda()
    class_indexes = []
    for key in y.keys():
        idx = y[key]
        if len(idx) == 2:
            class_inter += torch.cosine_similarity(z[idx[0]], z[idx[1]], dim=0)
        if len(idx) == 1:
            class_inter += torch.Tensor([1]).cuda()
        if len(idx) > 2:
            cat_M = []
            for i in range(1,len(idx)):
                cat_M.append(z[idx[i]].unsqueeze(0))
                # print(z[idx[i]].unsqueeze(0).shape)
            cat_M = torch.cat(cat_M, dim=0)
            class_inter += get_att_dis(z[idx[0]].unsqueeze(0),cat_M).mean()
        if len(idx) > 0:
            class_indexes.append(key)
    if len(class_indexes) > 1:
        classes_out = []
        for index in class_indexes:
            classes_out.append(z[y[index][0]].unsqueeze(0))
        classes_outs = torch.cat(classes_out[1:], dim=0)
        class_outer += get_att_dis(classes_out[0],classes_outs).sum()
    return (class_outer-class_inter)/len(class_indexes)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)


NetDict = {
    "resnet101": models.resnet101(pretrained = False, num_classes = 2),
    "densenet161": models.densenet161(pretrained = False, num_classes=2),
    "inception_v3": models.inception_v3(pretrained = False, aux_logits=False, num_classes = 2),
}

if __name__ == "__main__":
    fold = 0
    transform_train = transforms.Compose([
        # transforms.RandomRotation((0,60)),
        transforms.RandomResizedCrop(image_size,scale=(0.6,1.0)),
        # transforms.RandomAffine(5,),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.1642,), (0.317,)), 
        # Noise(),
    ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size,image_size)),
            transforms.Normalize((0.1642,), (0.317,)), 
        ])    
    for fold in range(1,5):
        trainset = datasets.ImageFolder(os.path.join('./data/withoutMOT/sup', 'train_fold'+str(fold),),transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0) 



        testset = datasets.ImageFolder(os.path.join('./data/withoutMOT/sup', 'val_fold'+str(fold)),transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

        for net_name in NetDict.keys():
            net = NetDict[net_name].cuda()
            net.load_state_dict(torch.load(os.path.join('./CNN_pretrained',net_name+'.pth')), False)
            if not os.path.exists(net_name):
                os.mkdir(net_name)
            if not os.path.exists(os.path.join(net_name,'fold'+str(fold))):
                os.mkdir(os.path.join(net_name,'fold'+str(fold)))        
            net_name = os.path.join(net_name,'fold'+str(fold))
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4) 
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            writer = SummaryWriter(net_name+'/logs')
            print("Start Training!")
            step = 0
            
            with open(net_name+"/acc.txt", "w") as f:
                for epoch in range(pre_epoch, EPOCH):
                    print(net_name+'\nEpoch: %d' % (epoch + 1))
                    net.train()
                    correct = 0
                    total = 0
                    correct = float(correct)
                    total = float(total)
                    for i, (inputs, labels) in enumerate(trainloader):
                        start_time = datetime.datetime.now()
                        batch = len(inputs)
                        inputs = inputs.cuda()
                        labels = labels.cuda() 
                        optimizer.zero_grad()
                        # forward + backward
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        writer.add_scalar('loss', loss, step)
                        writer.add_scalar('lr',optimizer.state_dict()['param_groups'][0]['lr'],step)
                        loss.backward()
                        optimizer.step()
                        end_time = datetime.datetime.now()
                        print('time:',end_time-start_time)
                        step += 1
                        print('batch:%d/%d, loss:%.4f, epoch:%d.'%(i,len(trainloader),loss, epoch))
                    scheduler.step()
                    # print('now the epoch is ',epoch)
                    # print('now the loss is ',loss)
                    trues = []
                    preds = []
                    if True:
                        print("Waiting Test!")
                        with torch.no_grad():

                            for i, (images, labels)  in enumerate(testloader):
                                net.eval()
                                images, labels = images.cuda(), labels.cuda()
                                outputs = net(images)
                                predicted = torch.argmax(outputs, 1)
                                trues.append(labels)
                                preds.append(predicted)

                            trues = torch.cat(trues,dim=0).cpu().numpy()
                            preds = torch.cat(preds,dim=0).cpu().numpy()
                            classify_result = metrics.compute_prec_recal(preds,trues)
                            print('EPOCH:{:5d},Acc:{:.3f},Precision:{:.3f},Recall:{:.3f},F1:{:.3f}.\n'.format(epoch + 1,classify_result[0],classify_result[1],classify_result[2],classify_result[3]))
                            if epoch % 5 == 0:
                                print('Saving model......')
                                if not os.path.exists(net_name+'/model'):
                                    os.mkdir(net_name+'/model')
                                torch.save(net.state_dict(), '%s/net_%03d.pth' % (net_name+'/model', epoch + 1))
                            f.write('EPOCH:{:5d},Acc:{:.3f},Precision:{:.3f},Recall:{:.3f},F1:{:.3f}.\n'.format(epoch + 1,classify_result[0],classify_result[1],classify_result[2],classify_result[3]))
                            f.flush()
            
    print("Training Finished!!!")