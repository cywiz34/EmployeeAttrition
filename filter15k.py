import csv

f1 = file('Verteego.csv', 'r')
c1 = csv.reader(f1)
list1=list(c1)
with open('data_15k.csv', 'a+') as filec:
	fieldnames = ['name','satisfaction_level','last_evaluation','number_projects','average_monthly_hours','time_spent_company','work_accident','left','promotion_last_5_years','department','salary','salary_level']
	writer = csv.DictWriter(filec, fieldnames=fieldnames)
	writer.writeheader()
	for i in list1[1:]:
		row=i[0].split(',')
		writer.writerow({'name':row[0],'satisfaction_level':float(row[1]),'last_evaluation':float(row[2]),'number_projects':int(row[3]),'average_monthly_hours':int(row[4]),'time_spent_company':int(row[5]),'work_accident':int(row[6]),'left':int(row[7]),'promotion_last_5_years':int(row[8]),'department':row[9],'salary':row[10],'salary_level':int(row[11])})
		


#name,satisfaction_level,last_evaluation,number_projects,average_monthly_hours,time_spent_company,work_accident,left,promotion_last_5_years,department,salary,salary_level
