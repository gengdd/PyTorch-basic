import torch
from torch.autograd import Variable
import numpy as np 
import torch.nn.functional as F

xy=np.loadtxt('C:/Users/GDD/Desktop/pytorch-basic/pytorch/data/diabetes.csv.gz',dtype=float,delimiter=',')

# print(xy.shape)
# print(xy[0,:])

# x_data=Variable(torch.from_numpy(xy[:,0:-1]))
# y_data=Variable(torch.from_numpy(xy[:,-1]))

x_data=Variable(torch.Tensor(xy[:,0:-1]))
y_data=Variable(torch.Tensor(xy[:,[-1]]))
# y_data=Variable(torch.Tensor(xy[:,-1][:,np.newaxis]))

# print(x_data.data.shape)
# print(y_data.shape)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1=torch.nn.Linear(8,6)
        self.l2=torch.nn.Linear(6,4)
        self.l3=torch.nn.Linear(4,1)

        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,x):
        out1=self.sigmoid(self.l1(x))
        out2=self.sigmoid(self.l2(out1))
        y_pred=self.sigmoid(self.l3(out2))
        return y_pred

model=Model()
#loss funcation
criterion=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(100):
    y_pred=model(x_data)
    
    loss=criterion(y_pred,y_data)
    print(epoch,loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y_pred=model(x_data)
print(y_pred.data.numpy())
print(y_data.data.numpy())

'''
#快速搭建法
model=torch.nn.Sequential(
    torch.nn.Linear(8,6),
    torch.nn.ReLU(),
    torch.nn.Linear(6,4),
    torch.nn.ReLU(),
    torch.nn.Linear(4,1),
    torch.nn.Sigmoid()
)

loss_func=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.02)

for epoch in range(200):
    y_pred=model(x_data)
    loss=loss_func(y_pred,y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(y_pred.data.numpy())
'''