import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x_data=Variable(torch.Tensor([[1.0],[2.0],[3.0]]))
y_data=Variable(torch.Tensor([[2.0],[4.0],[6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear=torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred

model=Model()

criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

'''
for epoch in range(500):
    y_pred=model(x_data)

    loss=criterion(y_pred,y_data)
    print(epoch,loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hour_var=Variable(torch.Tensor([[4.0]]))
y_pred=model(hour_var)
print('predcit:',4,model(hour_var).data[0][0])
'''

plt.ion()
plt.show()

for epoch in range(500):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%10==0:
        plt.cla()
        plt.scatter(x_data.data.numpy(),y_data.data.numpy())
        plt.plot(x_data.numpy(),y_pred.data.numpy(),'r-',lw=5)
        plt.text(2,2,'Loss=%.4f'%loss.data[0],fontdict={'size':20,'color':'red'})
        plt.pause(1)
plt.ioff()
plt.show()
