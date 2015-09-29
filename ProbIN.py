from numpy import * 
import numpy as np



def loadProbINdataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t') # split line to array based on tab delimiter
		fltLine = map(float,curLine) # cast data type to float
		dataMat.append(fltLine) # append list to a matrix
	return dataMat

def classifyMotionLabel(motionVector):
	Label_A_vector = [0.789634148975 , 0.12110714795 , 0.529785118975]
	Label_B_vector = [0.0347195935 , 0.98431831725 , 0.07812261]
	Label_C_vector = [0.793934286778 , 0.222574940889 , 0.668301732667]
	Label_D_vector = [0.566937970667 , 0.0412509603333 , 0.294235956167]
	Label_E_vector = [0.7024092805 , 0.46697209225 , 0.93441621425]
	Label_F_vector = [0.0904263145122 , 0.914914704049 , 0.400450085171]
	Label_G_vector = [0.745033689463 , 0.0712748704627 , 0.391802630716]
	Label_H_vector = [0.312025669667 , 0.763264765667 , 0.479369275667]
	Label_I_vector = [0.585725103 , 0.223192200286 , 0.734340564]
	Label_J_vector = [0.710627144913 , 0.113091247 , 0.508802911217]
	Label_K_vector = [0.839471456333 , 0.385856421 , 0.983790752333]
	Label_L_vector = [0.957817317 , 0.3012502055 , 0.7800295435]

	min_distance = sqrt(sum(power( (motionVector - Label_A_vector) , 2)))
	motLabel = "M1"

	if ( min_distance > sqrt(sum(power(motionVector - Label_B_vector, 2))) ):
		motLabel = "M2"
		min_distance = sqrt(sum(power(motionVector - Label_B_vector, 2)))

	if ( min_distance > sqrt(sum(power(motionVector - Label_C_vector, 2))) ):
		motLabel = "M3"
		min_distance = sqrt(sum(power(motionVector - Label_C_vector, 2)))

	if ( min_distance > sqrt(sum(power(motionVector - Label_D_vector, 2))) ):
		motLabel = "M4"
		min_distance = sqrt(sum(power(motionVector - Label_D_vector, 2)))

	if ( min_distance > sqrt(sum(power(motionVector - Label_E_vector, 2))) ):
		motLabel = "M5"
		min_distance = sqrt(sum(power(motionVector - Label_E_vector, 2)))

	if ( min_distance > sqrt(sum(power(motionVector - Label_F_vector, 2))) ):
		motLabel = "M6"
		min_distance = sqrt(sum(power(motionVector - Label_F_vector, 2)))

	if ( min_distance > sqrt(sum(power(motionVector - Label_G_vector, 2))) ):
		motLabel = "M7"
		min_distance = sqrt(sum(power(motionVector - Label_G_vector, 2)))	

	if ( min_distance > sqrt(sum(power(motionVector - Label_H_vector, 2))) ):
		motLabel = "M8"
		min_distance = sqrt(sum(power(motionVector - Label_H_vector, 2)))

	if ( min_distance > sqrt(sum(power(motionVector - Label_I_vector, 2))) ):
		motLabel = "M9"
		min_distance = sqrt(sum(power(motionVector - Label_I_vector, 2)))

	if ( min_distance > sqrt(sum(power(motionVector - Label_J_vector, 2))) ):
		motLabel = "M10"
		min_distance = sqrt(sum(power(motionVector - Label_J_vector, 2)))

	if ( min_distance > sqrt(sum(power(motionVector - Label_K_vector, 2))) ):
		motLabel = "M11"
		min_distance = sqrt(sum(power(motionVector - Label_K_vector, 2)))		

	if ( min_distance > sqrt(sum(power(motionVector - Label_L_vector, 2))) ):
		motLabel = "M12"
		min_distance = sqrt(sum(power(motionVector - Label_L_vector, 2)))


	return motLabel
	

def classifyDispLabel(dispVector):
	Label_A_vector = [0.47949965, 0]
	Label_B_vector = [0.24391939, 0.65005331]
	Label_C_vector = [0.65160991, 1]
	Label_D_vector = [0.66235972, 0.59802129]
	Label_E_vector = [0.61647991, 0.53295326]
	Label_F_vector = [0.64551821, 0.67475389]
	Label_G_vector = [0, 0.3630064]
	dispLabel = "D1" 

	min_distance = sqrt(sum(power( (dispVector  - Label_A_vector), 2 )))

	if ( min_distance > sqrt(sum(power(dispVector - Label_B_vector, 2))) ):
		dispLabel = "D2"
		min_distance = sqrt(sum(power(dispVector - Label_B_vector, 2)))

	if ( min_distance > sqrt(sum(power(dispVector - Label_C_vector, 2))) ):
		dispLabel = "D3"
		min_distance = sqrt(sum(power(dispVector - Label_C_vector, 2)))

	if ( min_distance > sqrt(sum(power(dispVector - Label_D_vector, 2))) ):
		dispLabel = "D4"
		min_distance = sqrt(sum(power(dispVector - Label_D_vector, 2)))

	if ( min_distance > sqrt(sum(power(dispVector - Label_E_vector, 2))) ):
		dispLabel = "D5"
		min_distance = sqrt(sum(power(dispVector - Label_E_vector, 2)))

	if ( min_distance > sqrt(sum(power(dispVector - Label_F_vector, 2))) ):
		dispLabel = "D6"
		min_distance = sqrt(sum(power(dispVector - Label_F_vector, 2)))

	if ( min_distance > sqrt(sum(power(dispVector - Label_G_vector, 2))) ):
		dispLabel = "D7"
		min_distance = sqrt(sum(power(dispVector - Label_G_vector, 2)))	

	return dispLabel

# calculate the probability given M label and D label; the function returns the count 
#	instead of the probability; The count goes to the Translation Model dictionary,
#	the dictionary has MD pair as the key and the corresponding count as values 
def calculate_P_m_d(MD_pair_fileName, motionLabel, dispLabel):
	# load the motion label pair data
	MD_pair_dataMat = []
	fr = open(MD_pair_fileName)
	for line in fr.readlines():
		curLine = line.strip().split(' \t') # split line to array based on tab delimiter
	
		MD_pair_dataMat.append(curLine) # append list to a matrix
	# now start calculating the probability
	count = 0
	for i in range(150):
		if (MD_pair_dataMat[i] == [motionLabel, dispLabel]):
			count = count + 1

	return count 


def sequenceSearch(inputList, targetSeq):
	count = 0
	for i in range(len(inputList)):
		if (inputList[i:i+len(targetSeq)] == targetSeq): 
			count= count+1
	return count

# calculate the probability of given D label and the n-gram trajectory length 
#	the function returns the count  instead of the probability; 
#   The count goes to the trajectory Model dictionary,
#	the dictionary has n-gram trajectory as the key and the corresponding count as values 
# 	dispLabelSeq = count di-n-1 ~ di
#   dispLabelSeq_2 = count di-n-1 ~ di-1
def calculate_P_d(MD_pair_fileName, dispLabelSeq, dispLabelSeq_2, SeqSearch = sequenceSearch):
	# load the motion label pair data
	MD_pair_dataMat = []
	fr = open(MD_pair_fileName)
	for line in fr.readlines():
		curLine = line.strip().split(' \t') # split line to array based on tab delimiter	
		MD_pair_dataMat.append(curLine) # append list to a matrix

	Disp_list = MD_pair_dataMat[:,1].A #the all the disp labels as a list

	count = 0

	count = SeqSearch(Disp_list, dispLabelSeq)

	count2 = 0 

	count2 = SeqSearch(Disp_list, dispLabelSeq_2)
	
	return count, count2





