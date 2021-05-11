import random
import numpy as np
import torch
from math import floor
import os
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import torch
import torch.utils.data as Data

def read_pdb(filename):
	
	with open(filename, 'r') as file:
		strline_L = file.readlines()
		# print(strline_L)

	X_list = list()
	Y_list = list()
	Z_list = list()
	atomtype_list = list()
	for strline in strline_L:
		# removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
		stripped_line = strline.strip()

		line_length = len(stripped_line)
		# print("Line length:{}".format(line_length))
		if line_length < 78:
			print("ERROR: line length is different. Expected>=78, current={}".format(line_length))
		
		X_list.append(float(stripped_line[30:38].strip()))
		Y_list.append(float(stripped_line[38:46].strip()))
		Z_list.append(float(stripped_line[46:54].strip()))

		atomtype = stripped_line[76:78].strip()
		if atomtype == 'C':
			atomtype_list.append('h') # 'h' means hydrophobic
		else:
			atomtype_list.append('p') # 'p' means polar

	return X_list, Y_list, Z_list, atomtype_list

def create_data_rec(pro, lig, label,
                    pro_list, lig_list, label_list, width, unit):
    file_name = "{}_{}_cg.pdb".format(("0000" + str(lig))[-4:], "lig")
    X_list, Y_list, Z_list, atomtype_list = read_pdb(DATA_DIR + file_name)
    x0 = floor(np.mean(X_list))
    y0 = floor(np.mean(Y_list))
    z0 = floor(np.mean(Z_list))
    all_atoms = list(zip(X_list, Y_list, Z_list, atomtype_list))
    use_atoms = [x for x in all_atoms if point_within_vox((x[0], x[1], x[2]),
                                                          x0 - width, x0 + width, y0 - width,
                                                          y0 + width, z0 - width, z0 + width)]
    
    lig_list.append(convert_xyz_to_vox(use_atoms, x0 - width, x0 + width, y0 - width,
                                       y0 + width, z0 - width, z0 + width, unit))
    
    
    
    file_name = "{}_{}_cg.pdb".format(("0000" + str(pro))[-4:], "pro")
    X_list, Y_list, Z_list, atomtype_list = read_pdb(DATA_DIR + file_name)
    all_atoms = list(zip(X_list, Y_list, Z_list, atomtype_list))
    use_atoms = [x for x in all_atoms if point_within_vox((x[0], x[1], x[2]),
                                                          x0 - width, x0 + width, y0 - width,
                                                          y0 + width, z0 - width, z0 + width)]
    pro_list.append(convert_xyz_to_vox(use_atoms, x0 - width, x0 + width, y0 - width,
                                       y0 + width, z0 - width, z0 + width, unit))
    label_list.append(label)

def point_within_vox(pt, x_lower=-50, x_upper=50, y_lower=-50,
                     y_upper=50, z_lower=-50, z_upper=50):
    (atom_x, atom_y, atom_z) = pt
    check_x = (atom_x >= x_lower) and (atom_x < x_upper)
    check_y = (atom_y >= y_lower) and (atom_y < y_upper)
    check_z = (atom_z >= z_lower) and (atom_z < z_upper)
    return check_x and check_y and check_z


def convert_xyz_to_vox(atoms_list,
                       x_lower=-50, x_upper=50, y_lower=-50,
                       y_upper=50, z_lower=-50, z_upper=50, unit=1):
    length = int((x_upper-x_lower)/unit)
    vox = np.zeros((length, length, length, 2))

    for atom in atoms_list:
        (x, y, z, t) = atom
        index_x = floor((x-x_lower)/unit)
        index_y = floor((y-y_lower)/unit)
        index_z = floor((z-z_lower)/unit)
        if t =='h':
            vox[index_x,index_y,index_z, 0] += 1
        elif t =='p':
            vox[index_x, index_y, index_z, 1] += 1

    return vox

def PrepareTrainData(x,y):
    traindata=[]
    for i in range(0,len(x)):
        traindata.append([x[i],y[i]])
    return traindata

def Train(epoch,learning_rate,version):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    epochs=epoch

    epoch_list=[]
    train_loss_list=[]
    val_loss_list=[]
    train_acc_list=[]
    val_acc_list=[]

    best_loss=0

    for e in range(epochs):

        train_loss = 0.0

        train_total = 0
        val_total = 0
        
        train_correct = 0
        val_correct = 0

        for data, labels in train_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            target = model(data)
            loss = loss_func(target,labels.long())
            loss.backward()
            optimizer.step()
            train_loss = loss.item() * data.size(0)

            scores, predictions = torch.max(target.data, 1)
            train_total += labels.size(0)
            train_correct += int(sum(predictions == labels))
            train_acc = round(train_correct / train_total, 2)
        
        valid_loss = 0.0
        ##model.eval()     # Optional when not using Model Specific layer
        for data, labels in val_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            target = model(data)
            loss = loss_func(target,labels.long())
            valid_loss = loss.item() * data.size(0) 

            scores, predictions = torch.max(target.data, 1)
            val_total += labels.size(0)
            val_correct += int(sum(predictions == labels))
            val_acc = round(val_correct / val_total, 2)
        
        train_loss=train_loss / len(train_loader)
        valid_loss=valid_loss / len(val_loader)
        print('Epoch {} \t\t Training Loss: {} \t\t Validation Loss: {} \t\t Training Accuracy: {} \t\t Validation Accuracy: {}'.format(e+1,train_loss,valid_loss,train_acc,val_acc))
        epoch_list.append(e+1)
        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)


        PATH='model_{}_epoch_{}.pkl'.format(version,e+1)
        torch.save(model, PATH)   
    return model,pd.DataFrame({'Epoch':epoch_list,'Training Loss':train_loss_list,'Validation Loss':val_loss_list,'Training Accuracy':train_acc_list,'Validation Accuracy':val_acc_list})

def Plot(data,lr,cols):
    value_col=cols

    num1=1
    num2=0.37
    num3=3
    num4=0

    plt.figure(figsize=(10,7))
    for h in value_col:
        ##plt.plot(datelist, data[h],color='r', lw=1, marker='s', ms=4, label=h,linewidth=2.5)
        plt.plot(data['Epoch'], data[h], lw=1, marker='s', ms=4, label=h)
        plt.xlabel('Epoch',fontsize=15)  
        plt.ylabel('Loss',fontsize=15)   
    plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    plt.title("Loss when Learning Rate = "+str(lr))
    plt.show()

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.cnn_layers = torch.nn.Sequential(
            # Defining a 3D convolution layer
            torch.nn.Conv3d(4, 30, kernel_size=(3,3,3), stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=(2,2,2), stride=1),
            torch.nn.BatchNorm3d(30),
            
            # Defining a 3D convolution layer
            torch.nn.Conv3d(30, 60, kernel_size=(3,3,3), stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=(2,2,2), stride=1),
            torch.nn.BatchNorm3d(60),
            
            # Defining a 3D convolution layer
            torch.nn.Conv3d(60, 90, kernel_size=(3,3,3), stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=(2,2,2), stride=1),
            torch.nn.BatchNorm3d(90),
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(442170, 256),
            torch.nn.Dropout(0.4),
            torch.nn.ReLU(),
            torch.nn.Linear(256,2),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)

        ##Flatten
        x = x.view(x.size(0), -1)

        x = self.linear_layers(x)
        return x

if __name__=='__main__':

    DATA_DIR='training_data/'

    ##---------- Prepare training set and validation set------------##
    unit = 1
    width = 10
    length = int(2 * width / unit)
    num_test = 200


    total_idx = [x+1 for x in list(range(3000))]
        # set seed so that every model will use the same test set
    np.random.seed(7)
    test_idx = [x+1 for x in np.random.choice(3000, num_test, replace=False)]
    train_idx = list(set(total_idx).difference(set(test_idx)))

    pro_train = []
    lig_train = []
    label_train = []

    for idx in train_idx:
        
        create_data_rec(idx,idx,1,pro_train,lig_train,label_train,width,unit)
        idx_false1 = random.choice(list(set(train_idx) - set([idx])))
            
        create_data_rec(idx,idx_false1,0,pro_train,lig_train,label_train,width,unit)
        #idx_false2 = random.choice(list(set(train_idx) - set([idx, idx_false1])))
            
        #create_data_rec(idx,idx_false2,0,pro_train,lig_train,label_train,width,unit)

    pro_train = np.array(pro_train)
    lig_train = np.array(lig_train)

    X = np.concatenate([pro_train, lig_train], axis=4)
    y = np.array(label_train)

    ## Reshape X
    X=np.swapaxes(X,1,4)
    X=np.swapaxes(X,2,4)
    X=np.swapaxes(X,3,4)
    ## convert data to torch.tensor format
    x_train=torch.from_numpy(X[1120:]).float()
    x_val=torch.from_numpy(X[0:1120]).float()
    y_train=torch.from_numpy(y[1120:]).long()
    y_val=torch.from_numpy(y[0:1120]).long()

    ## Prepare Data loader
    train_data=PrepareTrainData(x_train,y_train)
    val_data=PrepareTrainData(x_val,y_val)

    train_loader = Data.DataLoader(
    dataset=train_data,      # torch TensorDataset format
    batch_size=20,      
    shuffle=True,              
    num_workers=2,              
    )

    val_loader = Data.DataLoader(
    dataset=val_data,      # torch TensorDataset format
    batch_size=20,      
    shuffle=True,              
    num_workers=2, 
    )

    ##-------------------------Train the model------------------------##
    model=Model()
    print(model)

    model,loss_df=Train(20,0.01,1)

    ## Plot loss & accuracy
    loss=['Training Loss','Validation Loss']
    acc=['Training Accuracy','Validation Accuracy']

    Plot(loss_df,0.01,loss)
    Plot(loss_df,0.01,acc)
