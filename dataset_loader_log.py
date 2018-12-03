import torch
import numpy as np 
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader

class DiabetesDataset(Dataset):
    def __init__(self):
        xy=np.loadtxt('C:/Users/GDD/Desktop/pytorch-basic/pytorch/data/diabetes.csv.gz',dtype=float,delimiter=',')
        self.len=xy.shape[0]
        self.x_data=torch.Tensor(xy[:,0:-1])
        self.y_data=torch.Tensor(xy[:,[-1]])

    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len

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


if __name__ == "__main__":
    
    dataset=DiabetesDataset()
    train_loader=DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    model=Model()
    loss_func=torch.nn.BCELoss(size_average=True) ##二分类交叉熵，前边加sigmoid
    optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

    for epoch in range(2):
        for i,data in enumerate(train_loader,0):
            inputs,labels=data
            inputs,labels=Variable(inputs),Variable(labels)

            y_pred=model(inputs)
            loss=loss_func(y_pred,labels)
            print(epoch,i,loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
