import torch
import numpy as np 
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data


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
        
if __name__ == "__main__":
    dataset=DiabetesDataset()
    train_loader=DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    for epoch in range(2):
        for i,data in enumerate(train_loader,0):
            inputs,labels=data

            inputs,labels=Variable(inputs),Variable(labels)

            print(epoch,i,'inputs:',inputs.data,'labels:',labels.data)
'''
class DiabetesDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        xy=np.loadtxt('C:/Users/GDD/Desktop/pytorch-basic/pytorch/data/diabetes.csv.gz',dtype=float,delimiter=',')
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Run your training process
        print(epoch, i, "inputs", inputs.data, "labels", labels.data)

if __name__=='__main__':

    BATCH_SIZE=5

    x=torch.linspace(1,10,10)
    y=torch.linspace(10,1,10)

    torch_dataset=Data.TensorDataset(x,y)
    loader=Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    for epoch in range(3):
        for step,(batch_x,batch_y) in enumerate(loader):
            print('Epoch:',epoch,'|Step:',step,'|batch x:',batch_x.numpy(),'| batch y:',batch_y.numpy())
'''