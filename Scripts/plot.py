import csv

f1 = file('Extracted.csv', 'r')
c1 = csv.reader(f1)
list1=list(c1)
jsy1=[0 for i in range(5)]
jsy2=[0 for i in range(5)]
jsy3=[0 for i in range(5)]
jsy4=[0 for i in range(5)]
jsn1=[0 for i in range(5)]
jsn2=[0 for i in range(5)]
jsn3=[0 for i in range(5)]
jsn4=[0 for i in range(5)]

for i in list1:
	if str(i[0])=='1':
		for j in xrange(1,6):	
			if i[j]=='1':
				jsy1[j-1]+=1
			if i[j]=='2':
				jsy2[j-1]+=1
			if i[j]=='3':
				jsy3[j-1]+=1
			if i[j]=='4':
				jsy4[j-1]+=1

	else:
		for j in xrange(1,6):
			if i[j]=='1':
				jsn1[j-1]+=1
			if i[j]=='2':
				jsn2[j-1]+=1
			if i[j]=='3':
				jsn3[j-1]+=1
			if i[j]=='4':
				jsn4[j-1]+=1

print jsy1,jsy2,jsy3,jsy4
print jsn1,jsn2,jsn3,jsn4