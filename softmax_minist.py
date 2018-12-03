from __future__ import print_function
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.autograd import Variable
from torchvision import datasets,transforms

batch_size=64

train_dataset=datasets.MNIST(
    root='C:/Users/GDD/Desktop/pytorch-basic/pytorch\mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=False
)
test_dataset=datasets.MNIST(
    root='C:/Users/GDD/Desktop/pytorch-basic/pytorch\mnist',
    train=False,
    transform=transforms.ToTensor()
)

train_loader=torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader=torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1=nn.Linear(784,520)
        self.l2=nn.Linear(520,320)
        self.l3=nn.Linear(320,240)
        self.l4=nn.Linear(240,120)
        self.l5=nn.Linear(120,10)

    def forward(self,x):
        x=x.view(-1,784)
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=F.relu(self.l4(x))
        return self.l5(x)

model=Net()

loss_func=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=Variable(data),Variable(target)
        output=model(data)
        loss=loss_func(output,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx%10==0:
            print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),
                100.*batch_idx/len(train_loader),loss.data[0]
            ))

def test():
    model.eval()
    test_loss=0
    correct=0
    for data,target in test_loader:
        data,target=Variable(data,volatile=True),Variable(target)
        output=model(data)

        test_loss+=loss_func(output,target).data[0]
'''
c:\Users\GDD\Desktop\pytorch-basic\pytorch\softmax_minist.py:77: 
UserWarning: volatile was removed and now has no effect. 
Use `with torch.no_grad():` instead.data,target=Variable(data,volatile=True),Variable(target)
c:\Users\GDD\Desktop\pytorch-basic\pytorch\softmax_minist.py:80: 
UserWarning: invalid index of a 0-dim tensor. 
This will be an error in PyTorch 0.5. 
Use tensor.item() to convert a 0-dim tensor to a Python number
'''
        pred=output.data.max(1,keepdim=True)[1]
        correct+=pred.eq(target.data.view_as(pred)).cpu().sum()
    
    test_loss/=len(test_loader.dataset)
    print('\nTest set: Average loss:{:.4f},Accuracy:{}/{} ({:.0f}%)\n'.format(
        test_loss,correct,len(test_loader.dataset),
        100.*correct/len(test_loader.dataset)
    ))

for epoch in range(1,5):
    train(epoch)
    test()