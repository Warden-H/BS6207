import random
import numpy as np
import torch
from math import floor
import os
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd

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

def get_pred_df_for_pro(pro, lig_list, model, width, unit):
    pro_test, lig_test, label_test = [], [], []
    for lig in lig_list:
        create_data_rec2(pro, lig, -1, pro_test, lig_test, label_test, width, unit)
    testset = np.concatenate([pro_test, lig_test], axis=4)
    
    testset=np.swapaxes(testset,1,4)
    testset=np.swapaxes(testset,2,4)
    testset=np.swapaxes(testset,3,4)
    
    pred = model2(torch.tensor(testset).float())
    softmax=torch.nn.Softmax()
    pred = softmax(pred)[:,1]
    d = {'idx': lig_list, 'prob': list(pred.data.numpy())}
    pred_top20 = pd.DataFrame(d).sort_values(by='prob', ascending=False).head(20).set_index('idx')
    return pred_top20

def get_pred_df_for_pro_no_softmax(pro, lig_list, model, width, unit):
    pro_test, lig_test, label_test = [], [], []
    for lig in lig_list:
        create_data_rec2(pro, lig, -1, pro_test, lig_test, label_test, width, unit)
    testset = np.concatenate([pro_test, lig_test], axis=4)
    
    testset=np.swapaxes(testset,1,4)
    testset=np.swapaxes(testset,2,4)
    testset=np.swapaxes(testset,3,4)
    
    pred = model2(torch.tensor(testset).float())
    #softmax=torch.nn.Softmax()
    #pred = softmax(pred)[:,1]
    d = {'idx': lig_list, 'prob': list(pred.data.numpy()[:,1])}
    pred_top20 = pd.DataFrame(d).sort_values(by='prob', ascending=False).head(20).set_index('idx')
    return pred_top20

def create_data_rec2(pro, lig, label,
                    pro_list, lig_list, label_list, width, unit):
    file_name = "{}_{}_cg.pdb".format(("0000" + str(lig))[-4:], "lig")
    X_list, Y_list, Z_list, atomtype_list = read_pdb_test(DATA_DIR2+ file_name)
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
    X_list, Y_list, Z_list, atomtype_list = read_pdb_test(DATA_DIR2 + file_name)
    all_atoms = list(zip(X_list, Y_list, Z_list, atomtype_list))
    use_atoms = [x for x in all_atoms if point_within_vox((x[0], x[1], x[2]),
                                                          x0 - width, x0 + width, y0 - width,
                                                          y0 + width, z0 - width, z0 + width)]
    pro_list.append(convert_xyz_to_vox(use_atoms, x0 - width, x0 + width, y0 - width,
                                       y0 + width, z0 - width, z0 + width, unit))
    label_list.append(label)

def read_pdb_test(filename):
	
	with open(filename, 'r') as file:
		strline_L = file.readlines()
		# print(strline_L)

	X_list = list()
	Y_list = list()
	Z_list = list()
	atomtype_list = list()
	for strline in strline_L:

		stripped_line = strline.split()

		line_length = len(stripped_line)
		
		
		X_list.append(float(stripped_line[0]))
		Y_list.append(float(stripped_line[1]))
		Z_list.append(float(stripped_line[2]))

		atomtype = stripped_line[3]
		if atomtype == 'C':
			atomtype_list.append('h') # 'h' means hydrophobic
		else:
			atomtype_list.append('p') # 'p' means polar

	return X_list, Y_list, Z_list, atomtype_list

def OutputFile(top10_list):
    np_top10_list=np.array(top10_list)
    test_prediction=pd.DataFrame({'pro_id':np_top10_list[:,0]})
    for i in range(1,11):
        test_prediction['lig{}_id'.format(i)]=np_top10_list[:,i]
    return test_prediction

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
    ## Global Variable
    unit = 1
    width = 10
    length = int(2 * width / unit)
    num_test = 200
    DATA_DIR='training_data/'
    ## Use model generated at epoch 2
    model2=torch.load('model_1_epoch_2.pkl')
    DATA_DIR2='testing_data_release/testing_data/'

    num_test =825

    ##===================Produce Result using the loaded model(with softmax transformed probability)=========================
    top10_list = []
    lig_list = list(range(1, num_test))

    for pro in range(1, num_test):
        if pro % 10 == 0:
            print("Trying to match for pro_{}".format(pro))
        pred1_top20 = get_pred_df_for_pro(pro, lig_list, model2, width, unit)

        pred_top20 = pred1_top20
        top10_idx = pred_top20.sort_values(by=['prob'],
                                        ascending=[False]).head(10).index
        print(pro, list(top10_idx))
        top10_list.append([pro] + list(top10_idx))
    
    test_prediction=OutputFile(top10_list)
    test_prediction.to_csv('test_prediction.txt',sep='\t', index=False)


    ##===================Produce Result using the loaded model(without softmax transformed probability)=========================
    top10_list_no_softmax = []
    lig_list = list(range(1, num_test))
    ## Protein 1~264
    for pro in range(1, 265):
        if pro % 10 == 0:
            print("Trying to match for pro_{}".format(pro))
        pred1_top20 = get_pred_df_for_pro_no_softmax(pro, lig_list, model2, width, unit)

        pred_top20 = pred1_top20
        top10_idx = pred_top20.sort_values(by=['prob'],
                                        ascending=[False]).head(10).index
        print(pro, list(top10_idx))
        top10_list_no_softmax.append([pro] + list(top10_idx))
    
    test_prediction_no_softmax=OutputFile(top10_list_no_softmax)
    test_prediction_no_softmax.to_csv('test_prediction_no_softmax.txt',sep='\t', index=False)

    ## Protein 265~824
    top10_list_no_softmax = []
    lig_list = list(range(1, num_test))

    for pro in range(265, num_test):
        if pro % 10 == 0:
            print("Trying to match for pro_{}".format(pro))
        pred1_top20 = get_pred_df_for_pro_no_softmax(pro, lig_list, model2, width, unit)

        pred_top20 = pred1_top20
        top10_idx = pred_top20.sort_values(by=['prob'],
                                        ascending=[False]).head(10).index
        print(pro, list(top10_idx))
        top10_list_no_softmax.append([pro] + list(top10_idx))
