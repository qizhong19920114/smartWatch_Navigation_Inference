import kMeans
import ProbIN
from numpy import *
import subprocess

import numpy as np

# 7 centroids for the Training sensor data
# centlist:  [matrix([[ 0.74017507,  0.35886748,  0.65792925,  0.09850031,  0.56319411,
#           0.40595958]]), matrix([[ 0.70064452,  0.470423  ,  0.72913838,  0.52446257,  0.50978768,
#           0.42553676]]), matrix([[ 0.73391835,  0.25733648,  0.48284854,  0.35589827,  0.61996549,
#           0.37411704]]), matrix([[ 0.45630783,  0.7068354 ,  0.54165635,  0.72202249,  0.38487588,
#           0.51708114]]), matrix([[ 0.7359613 ,  0.37524037,  0.67209691,  0.67286115,  0.55544224,
#           0.4089276 ]]), matrix([[ 0.71862613,  0.39010584,  0.68409843,  0.90080511,  0.54869346,
#           0.41495983]]), matrix([[ 0.72298071,  0.2547888 ,  0.50125114,  0.72605772,  0.62117875,
#           0.38394224]])]

# 7 centroids for the Training displacement data 
# centlist:  [matrix([[-1.65285861,  0.28437462]]), matrix([[ 0.68061024, -0.02348856]]), matrix([[-0.42519685, -0.91214178]]), matrix([[-1.16255171, -0.49227318]]), matrix([[ -6.86368201e-06,  -2.09717778e-03]]), matrix([[ 0.02382395, -1.35331825]]), matrix([[-0.01220834, -0.79834503]])]



#datMat = mat(kMeans.loadDataSet('gps_data_wsu_to_steakhouse_1Hz_delta_nom_2_3.txt'))

# print min(datMat[:,0])
# print min(datMat[:,1])
# print max(datMat[:,1])
# print max(datMat[:,0])

#print "randCent: "

#print kMeans.randCent(datMat,5)

# 5 centroids: accelerate, decelerate, constant_speed, turning left
# turning right. 
#kMeans.biKmeans(datMat,7)

# print centList

# disp_label = ProbIN.classifyMotionLabel(np.array([0.74017507,  0.300000,  0.65792925,  0.09850031,  0.56319411, 0.40595958])) 

# print disp_label

#Create two dictionareis for motion labels and disp_Label 





motionLabelDic = \
{'M1': [0.789634148975 , 0.12110714795 , 0.529785118975], \
'M2': [0.0347195935 , 0.98431831725 , 0.07812261], \
'M3': [0.793934286778 , 0.222574940889 , 0.668301732667], \
'M4': [0.566937970667 , 0.0412509603333 , 0.294235956167], \
'M5': [0.7024092805 , 0.46697209225 , 0.93441621425], \
'M6': [0.0904263145122 , 0.914914704049 , 0.400450085171], \
'M7': [0.745033689463 , 0.0712748704627 , 0.391802630716],\
'M8': [0.312025669667 , 0.763264765667 , 0.479369275667],\
'M9': [0.585725103 , 0.223192200286 , 0.734340564],\
'M10': [0.710627144913 , 0.113091247 , 0.508802911217],\
'M11': [0.839471456333 , 0.385856421 , 0.983790752333],\
'M12': [0.957817317 , 0.3012502055 , 0.7800295435]}

dispLabelDic = \
{'D1': [0.47949965, 0], \
'D2': [0.24391939, 0.65005331], \
'D3': [0.65160991, 1], \
'D4': [0.66235972, 0.59802129], \
'D5': [0.61647991, 0.53295326], \
'D6': [0.64551821, 0.67475389], \
'D7': [0, 0.3630064]} 


motionDataMat = mat(kMeans.loadDataSet('motionData_Training.txt'))
dispDataMat = mat(kMeans.loadDataSet('GPS_1Hz_training.txt'))

print motionDataMat[0]

#print ProbIN.classifyMotionLabel(motionDataMat[0]) == 'M4'

print dispDataMat[0]

print ProbIN.classifyDispLabel(dispDataMat[0])

f = open('MD_pair_1Hz_for_Training.txt', 'w')

for i in range(len(dispDataMat)):
	print >> f, ProbIN.classifyMotionLabel(motionDataMat[i]), "\t", ProbIN.classifyDispLabel(dispDataMat[i])

f.close()


motionDataMat_eval = mat(kMeans.loadDataSet('motionData_Eval.txt'))
f2 = open('M_Label_for_Eval.txt', 'w')
for i in range(len(motionDataMat_eval)):
	print >> f2, ProbIN.classifyMotionLabel(motionDataMat_eval[i])

f2.close()

motionDataMat_complete = mat(kMeans.loadDataSet('motionDataComplete.txt'))
dispDataMat_complete  = mat(kMeans.loadDataSet('GPS_1Hz_complete.txt'))
f_complete  = open('MD_pair_1Hz_complete.txt', 'w')

for i in range(len(dispDataMat_complete)):
	print >> f_complete, ProbIN.classifyMotionLabel(motionDataMat_complete[i]), "\t", ProbIN.classifyDispLabel(dispDataMat_complete[i])

f_complete.close()



#print ProbIN.calculate_P_m_d("MD_pair_1Hz.txt",'M2','D3')

#Now create two dictionareis: One for MD pair counts, the other for n-gram D counts

MD_label_mat = mat(kMeans.loadDataSet_string('MD_pair_1Hz_for_Training.txt'))
#print MD_label_mat

# Create Dictionareis for training and prediction
Dict_MDpair_cnt  = {}

for i in range(shape(MD_label_mat)[0]): 
	MD_pair_str = MD_label_mat[i].A[0][0]+ MD_label_mat[i].A[0][1]
	#print MD_pair_str
	if ( MD_pair_str in Dict_MDpair_cnt): 
	 	Dict_MDpair_cnt[MD_pair_str] +=1
	else: 
	 	Dict_MDpair_cnt[MD_pair_str] =1

print Dict_MDpair_cnt

Dict_n_gram_cnt = {}
Dict_n_minus_gram_cnt = {}

print MD_label_mat[:,1].A[:,0]

Disp_Label_array = MD_label_mat[:,1].A[:,0]

n = 5 # this is the n-gram length

for i in range(n, len(Disp_Label_array)):
	#print "".join(Disp_Label_array[i-n:i])
	n_gram_str = "".join(Disp_Label_array[i-n:i])
	n_minus_gram_str = "".join(Disp_Label_array[i-n:i-1])

	if ( n_gram_str in Dict_n_gram_cnt): 
	 	Dict_n_gram_cnt[n_gram_str] +=1
	else: 
	 	Dict_n_gram_cnt[n_gram_str] =1

	if ( n_minus_gram_str in Dict_n_minus_gram_cnt): 
	 	Dict_n_minus_gram_cnt[n_minus_gram_str] +=1
	else: 
	 	Dict_n_minus_gram_cnt[n_minus_gram_str] =1

print Dict_n_gram_cnt
print "\n"
print Dict_n_minus_gram_cnt

# Now two dicitonary are created. It doesn't seem like we need the search funciton in ProbIN module.

# Now we can start the training process. 

# for j in range(3):
# 	for i in range(n,shape(MD_label_mat)[0]): 
# 		print i


#Skip training: let's just try with the untrained model..

#COMMAND = "awk '{print $1}' MD_pair_1Hz_for_eval.txt > M_Label_eval.txt"  

#subprocess.call(COMMAND, shell=True)

M_label_list = [line.strip() for line in open("M_Label_for_Eval.txt", 'r')]

print M_label_list



f3 = open('predicted_D_labels_values_eval.txt', 'w')
f4 = open('predicted_D_labels_eval.txt', 'w')

initialDisp = 'D6'

for i in range(len(M_label_list)):
	#print M_label_list[i]
	max_cnt = 0
	max_str = ''
	max_str_label = ''

	if (M_label_list[i] + 'D1') in Dict_MDpair_cnt.keys():
		if(Dict_MDpair_cnt[M_label_list[i] + 'D1']> max_cnt):
			max_cnt = Dict_MDpair_cnt[M_label_list[i] + 'D1']
			max_str = dispLabelDic['D1'] 
			max_str_label = 'D1'

	if (M_label_list[i] + 'D2') in Dict_MDpair_cnt.keys():
		if(Dict_MDpair_cnt[M_label_list[i] + 'D2']> max_cnt):
			max_cnt = Dict_MDpair_cnt[M_label_list[i] + 'D2']
			max_str = dispLabelDic['D2'] 
			max_str_label = 'D2'

	if (M_label_list[i] + 'D3') in Dict_MDpair_cnt.keys():
		if(Dict_MDpair_cnt[M_label_list[i] + 'D3']> max_cnt):
			max_cnt = Dict_MDpair_cnt[M_label_list[i] + 'D3']
			max_str = dispLabelDic['D3'] 
			max_str_label = 'D3'

	if (M_label_list[i] + 'D4') in Dict_MDpair_cnt.keys():
		if(Dict_MDpair_cnt[M_label_list[i] + 'D4']> max_cnt):
			max_cnt = Dict_MDpair_cnt[M_label_list[i] + 'D4']
			max_str = dispLabelDic['D4'] 
			max_str_label = 'D4'

	if (M_label_list[i] + 'D5') in Dict_MDpair_cnt.keys():
		if(Dict_MDpair_cnt[M_label_list[i] + 'D5']> max_cnt):
			max_cnt = Dict_MDpair_cnt[M_label_list[i] + 'D5']
			max_str = dispLabelDic['D5'] 
			max_str_label = 'D5'

	if (M_label_list[i] + 'D6') in Dict_MDpair_cnt.keys():
		if(Dict_MDpair_cnt[M_label_list[i] + 'D6']> max_cnt):
			max_cnt = Dict_MDpair_cnt[M_label_list[i] + 'D6']
			max_str = dispLabelDic['D6'] 
			max_str_label = 'D6'

	if (M_label_list[i] + 'D7') in Dict_MDpair_cnt.keys():
		if(Dict_MDpair_cnt[M_label_list[i] + 'D7']> max_cnt):
			max_cnt = Dict_MDpair_cnt[M_label_list[i] + 'D7']
			max_str = dispLabelDic['D7'] 
			max_str_label = 'D7'

	print >> f3, max_str[0], "\t", max_str[1]
	print >> f4, max_str_label

f3.close()
f4.close()




Predicted_dispDataMat = mat(kMeans.loadDataSet('my_pred_values.txt'))

starting_lat = 37.67906


starting_lon = -97.33603



f_predicted_gps = open('GPS_trace.txt', 'w')

for i in range(shape(Predicted_dispDataMat)[0]):

	starting_lat = starting_lat + (Predicted_dispDataMat.A[i][0]*(0.0076+0.013985)-0.013985)

	starting_lon = starting_lon + (Predicted_dispDataMat.A[i][1]*(0.01122+0.00753999999999)-0.01122)

	print >> f_predicted_gps, starting_lat, ",", starting_lon


f_predicted_gps.close()























