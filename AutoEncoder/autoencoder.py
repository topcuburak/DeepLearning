#burak topcu 283078927
#ceng506 hw2

import torch
from torch import tensor
import torchvision
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset
import xlsxwriter  


class EPFLGIMS08Dataset(Dataset):					##for importing the epfl gims 08 dataset codes are given by the instructor
    def __init__(self, basedir, transforms=None,
                 seq_filter=None, is_test=False):
        self.basepath = Path(basedir)
        self.transforms = transforms
        self.seq_filter = set(seq_filter)
        self.is_test = is_test

        self.items = self._load_items()

    def _load_items(self):
        mdata = self._load_metadata()
        # print(mdata)
        basepath = self.basepath.joinpath('tripod-seq')
        image_format = mdata['image_format']
        bbox_format = mdata['bbox_format']
        items = []
        seq_list = set(range(mdata['n_seq']))
        if self.seq_filter is not None:
            seq_list = seq_list.intersection(self.seq_filter)
        for i in seq_list:
            bbox_path = basepath.joinpath(bbox_format % (i+1))
            with open(bbox_path) as fin:
                lines = [line.strip() for line in fin.readlines()]
                bboxes = [[round(float(i)) for i in line.split()]
                          for line in lines]

            seq_len = mdata['seq_len'][i]
            item_list = set(range(seq_len))
            test_ids = set(range(2, seq_len, 3))
            if self.is_test:
                item_list = test_ids
            else:
                item_list -= test_ids
            for j in item_list:
                filepath = basepath.joinpath(image_format % (i+1, j+1))
                tl_x = bboxes[j][0]
                tl_y = bboxes[j][1]
                br_x = tl_x + bboxes[j][2]
                br_y = tl_y + bboxes[j][3]
                item = (filepath, i-1, (tl_x, tl_y, br_x, br_y))
                items.append(item)
        return items

    def _load_metadata(self):
        metadata_path = self.basepath.joinpath('tripod-seq/tripod-seq.txt')
        with open(metadata_path, 'r') as fin:
            lines = fin.readlines()
            lines = [line.strip() for line in lines]
        mdata = {}
        mdata['n_seq'] = int(lines[0].split()[0])
        mdata['seq_len'] = [int(i) for i in lines[1].split()]
        mdata['image_format'] = lines[2]
        mdata['bbox_format'] = lines[3]
        mdata['rot_len'] = [int(i) for i in lines[4].split()]
        mdata['front_idx'] = [int(i) for i in lines[5].split()]
        mdata['is_cw'] = [int(i) > 0 for i in lines[6].split()]

        metadata_path = self.basepath.joinpath('tripod-seq/times.txt')
        with open(metadata_path, 'r') as fin:
            lines = [line.strip() for line in fin.readlines()]
        mdata['dt'] = [[int(i) for i in line.split()] for line in lines]
        return mdata

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        with open(item[0], 'rb') as fin:
            img = Image.open(fin)
            img = img.convert('RGB')
            img = img.crop(item[2])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, item[1]

def DataSequenceLoader(sequence, batch_size): #this function is used to import one sequence at a time
															##[1-10] for test, [11-15] for validation, [16-20] for test

    img_trans = transforms.Compose([torchvision.transforms.Resize((64,64)), transforms.ToTensor()])		##to resize the function as 64*64 pixels
    dataset = EPFLGIMS08Dataset('epfl-gims08', transforms=img_trans, seq_filter=[sequence])	#dataset is introduced with the given sequnce with reshaping
    train = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True)	#then shaped and loaded with the given batch size
    
    dataset_test = EPFLGIMS08Dataset('epfl-gims08', transforms=img_trans, seq_filter=[sequence], is_test=True)	#With the same manner, test set is loaded 
																									#by enabling is_test as True for the same sequence    
    test = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size, shuffle=True)
 
    print('Number of training {} items'.format(len(dataset)))
    print('Number of test {} items'.format(len(dataset_test)))

    return train, test, len(dataset), len(dataset_test)		#return train and test sets with their included number of samples


def DataAllLoader(data_seq, test_seq, valid_seq, batch_size_1, batch_size_2): #set(range(0,10))		#this function is used for importing multiple samples
																								#at the same time by combining them with respect to the batch sizes
    img_trans = transforms.Compose([torchvision.transforms.Resize((64,64)), transforms.ToTensor()])   ##to resize the function as 64*64 pixels

    dataset = EPFLGIMS08Dataset('epfl-gims08', transforms=img_trans, seq_filter=data_seq)  #dataset is introduced with the given sequnces with reshaping
    train = torch.utils.data.DataLoader(dataset,batch_size=batch_size_1, shuffle=True)    #then shaped and loaded with the given batch size

    dataset_valid = EPFLGIMS08Dataset('epfl-gims08', transforms=img_trans, seq_filter= test_seq, is_test=True)	#in the same manner, test and validation sets are 
    valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size_2, shuffle=True)	#loaded by enabling is_test as true 

    dataset_test = EPFLGIMS08Dataset('epfl-gims08', transforms=img_trans, seq_filter=valid_seq, is_test=True)
    test = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size_2, shuffle=True)

    print('Number of training {} items'.format(len(dataset)))
    print('Number of validation {} items'.format(len(dataset_valid)))
    print('Number of test {} items'.format(len(dataset_test)))

    return train, valid, test 		#return with train, validation and test sets

class AutoEncoder(nn.Module):		#auto encoder class that implements encoding and decoding functions with respect to the
    def __init__(self):				#given networks as in homework tutorial from section 7.2
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 5, stride=1), 	#encoder
						            nn.ReLU(inplace=True),
					    	        nn.Conv2d(32,32,5,stride=1),
					        	    nn.ReLU(inplace=True),
					            	nn.Conv2d(32,32,4,stride=2),
					        	    nn.ReLU(inplace=True),
					            	nn.Conv2d(32,32,3,stride=2),
					            	nn.ReLU(inplace=True),
					            	nn.Conv2d(32,8,4,stride=1))

        self.decoder = nn.Sequential(nn.ConvTranspose2d(8, 32, 4, stride=1),   #decoder
						            nn.ReLU(inplace=True),
						            nn.ConvTranspose2d(32,32,3,stride=2),
						            nn.ReLU(inplace=True),
						            nn.ConvTranspose2d(32,32,4,stride=2),
						            nn.ReLU(inplace=True),
						            nn.ConvTranspose2d(32,32,5,stride=1),
						            nn.ReLU(inplace=True),
						            nn.ConvTranspose2d(32,3,5,stride=1)
						        )

    def forward(self, x): 		#forward function which encodes and later decodes the given image
        return self.decoder(self.encoder(x))


def experiment_1():

	workbook = xlsxwriter.Workbook("Exp1_MSEloss.xlsx")	#to store MSE results
	sheet = workbook.add_worksheet()
	numb_of_epochs = 100		# number of epoch
	sequence = 0

	for i in range(20):		#at total, we have 20 sequences
		model = AutoEncoder()	#initialize AutoEncoder module to the model
		loss_funct = nn.MSELoss()	#initalize the MSEloss
		opt = torch.optim.Adam(model.parameters(), lr = 1e-3)	#optimizer used for the tune hyperparameters and model parameters
		batch_size = 8 	#batch size
		sequence = i                                             
		train_set, test_set, _, _ = DataSequenceLoader(sequence, batch_size) 	#Load data sequnce (0,1,2...19) with the batch size=8

		for epoch in range(numb_of_epochs): 	#run for 100 epochs for each of the sequence
			for train,_ in train_set: 	# train sample consist of 8 sample (identified with batch size)
				output = model.decoder(model.encoder(train))   #encode and decode
				loss = loss_funct(output, train)  #compute loss
				opt.zero_grad()		#gradient optimizer
				loss.backward()		#backword loss
				opt.step()		 #tune the parameters

		counter = 0
		sheet.write(0, sequence , "sequence_" + str(sequence)) 	#to write MSEresults
		batch_size = 1
		train_set, test_set, _, num_test = DataSequenceLoader(sequence, batch_size)  	#load test set as batch_size = 1 (sample by sample without combining)

		for test, test_lab in test_set:
			loss = loss_funct(test, model.forward(test))   #compute MSE loss
			counter += 1 
			sheet.write(counter, sequence, "{:.4f}".format(loss))  #record the mse loss with counter's row and sequnce's column

		batch_size = 8
		train_set, test_set, _, _ = DataSequenceLoader(sequence, batch_size)  #again load test set as batch_size = 8 to print 8 decoded images at the same time

		dataiter = iter(test_set) 	
		test_input, labels = dataiter.next()
		output = model(test_input)
		output = output.detach().numpy()

		for k in range(batch_size):  #plot each sample and encoded-decoded sample as batch size = 8
			plt.subplot(2,batch_size,k+1)
			plt.imshow(test_input[k][0])
			plt.subplot(2,batch_size,batch_size+k+1)
			plt.imshow(output[k][0])

		plt.savefig('DecodedSeq_'+str(sequence)+'.png') #save the output png for each sequence
	workbook.close() #close excel

def experiment_2():   #experiment 2 
	model = AutoEncoder()  #initialize model, loss function, optimizer, 
	loss_funct = nn.MSELoss()
	opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
	batch_size_1 = 8
	batch_size_2 = 1
	numb_of_epochs = 100

	data_seq = np.arange(0,10,1)   #initialize lists for training, validation and test
	valid_seq = np.arange(10,15,1)
	test_seq = np.arange(15,20,1)

	train_set, test_set, valid_set = DataAllLoader(data_seq, test_seq, valid_seq, batch_size_1, batch_size_2)  #load the required data
														# for training, test and validation sets, for training, bathc_size = 8 and for valid and test, batch size = 1 
	for epoch in range(numb_of_epochs):   #similar training as in experiment steps as in experiment 1
		for train, _ in train_set:
			output = model.decoder(model.encoder(train))
			loss = loss_funct(output, train)
			opt.zero_grad()
			loss.backward()
			opt.step()

		loss = 0
		for valid, _ in valid_set:	 # calculate loss by using the validation set after the each epoch and update the learning rates
			loss += loss_funct(valid, model.forward(valid))
													# as mentioned for the if blocks below
		loss_avg = loss / len(valid_set)
		print('epoch [{}/{}], loss:{:.4f}' .format(epoch + 1, numb_of_epochs, loss_avg))

		if loss_avg > 0.02 and epoch > 25:
			opt = torch.optim.Adam(model.parameters(), lr = 0.00075) 	#for example decrease it from 0.001 to 0.00075 if epoch is greater than 20 and 
																		#average loss is greater than 0.02
		if loss_avg > 0.013 and epoch > 50:
			opt = torch.optim.Adam(model.parameters(), lr = 0.0005)

		if loss_avg > 0.01 and epoch > 75:
			opt = torch.optim.Adam(model.parameters(), lr = 0.0002)

	workbook = xlsxwriter.Workbook("Exp2_MSEloss.xlsx")		#record the results 
	sheet = workbook.add_worksheet()
	batch_size = 8

	for i in range (0, 5):		#calculate the loss for each sequence in between [15,20]
		_, test_set, _, num_test = DataSequenceLoader(i+15, 1)
		sheet.write(0, i, "sequence_" + str(i+15))
		counter = 0
		for test, _ in test_set:		# calculate loss for each sample in the test set for sequences in between[15, 20]
			loss = loss_funct(test, model.forward(test))
			counter += 1 
			sheet.write(counter, i, "{:.4f}".format(loss))		#record the results
			print('test [{}/{}], loss:{:.4f}' .format(counter, num_test, loss))
		counter = 0

		_, test_set, _, num_test = DataSequenceLoader(i+15, 8)		# update the loaded sets as batch_size = 8 
		dataiter = iter(test_set)
		test_input, _ = dataiter.next()
		output = model(test_input)
		output = output.detach().numpy()

		for k in range(batch_size):		# plot the original and decoded 8 samples from the [15,20] sequences
			plt.subplot(2,batch_size,k+1)
			plt.imshow(test_input[k][0])
			plt.subplot(2,batch_size,batch_size+k+1)
			plt.imshow(output[k][0])

		plt.savefig('DecodedSeq_' + str(i) +'.png')		#save the plotted results and and close the excel
	workbook.close()

def interpolate(id1, id2):		#interpolation for the specified indexes

	model = AutoEncoder()		#initialize the models
	loss_funct = nn.MSELoss()
	opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
	batch_size = 8

	train_set, test_set, _, _ = DataSequenceLoader(4, 8)  #load from 4'th sequence with batch size = 8

	for epoch in range(100):  # train the model 100 number of epochs for the 4'th sequence 
		for train,_ in train_set:
			output = model.decoder(model.encoder(train))
			loss = loss_funct(output, train)
			opt.zero_grad()
			loss.backward()
			opt.step()
	_, test_set, _, _ = DataSequenceLoader(4, 1)    #re-load the 4'th sequence's test set

	counter = 0  			
	x1 = 0
	x2 = 0
	for test, _ in test_set:		# hold the indexed test samples from the test set
		if counter == id1:
			x1 = test
		elif counter == id2:
			x2 = test
			break
		counter += 1

	batch_size = 1 		#plot original samples of test set
	for k in range(batch_size):	
		plt.subplot(2,batch_size,k+1)
		plt.imshow(x1[k][0])
		plt.subplot(2,batch_size,batch_size+k+1)
		plt.imshow(x2[k][0])
	plt.savefig('org_picts.png')

	oT_x1 = model(x1) 		#encode and decode chosen samples 
	o_x1 = oT_x1.detach().numpy()
	oT_x2 = model(x2)
	o_x2 = oT_x2.detach().numpy()
	for k in range(batch_size): 	#plot them
		plt.subplot(2,batch_size,k+1)
		plt.imshow(o_x1[k][0])
		plt.subplot(2,batch_size,batch_size+k+1)
		plt.imshow(o_x2[k][0])
	plt.savefig('decoded_picts.png')

	ip_org = torch.lerp(x1, x2, 0.5).detach().numpy()  #interpolate orginal test samples
	ip_decoded = torch.lerp(oT_x1, oT_x2, 0.5).detach().numpy()   #interpolate orginal encoded-decoded samples

	for k in range(batch_size):  #plot trained samples
		plt.subplot(2,batch_size,k+1)
		plt.imshow(ip_org[k][0])
		plt.subplot(2,batch_size,batch_size+k+1)
		plt.imshow(ip_decoded[k][0])
	plt.savefig('interpolated_picts.png')

def experiment_3():
	interpolate(0, 1)   # 0th and 1st samples for the whatever sequence


'''
one can run each experiment by uncommenting the below sections
'''

#experiment_1()
#experiment_2()
#experiment_3()

