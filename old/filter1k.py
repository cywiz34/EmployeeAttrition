import csv

f1 = file('HR_Data.csv', 'r')
c1 = csv.reader(f1)
list1=list(c1)

with open('data_1k.csv', 'a+') as filec:
	fieldnames = ['Age','Attrition','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EducationField','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','Over18','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
	writer = csv.DictWriter(filec, fieldnames=fieldnames)
	writer.writeheader()

	for i in list1[1:]:
		#print i
		writer.writerow({'Age':int(i[0]),'Attrition':int(i[1]),'BusinessTravel':i[2],'DailyRate':int(i[3]),'Department':i[4],'DistanceFromHome':int(i[5]),'Education':int(i[6]),'EducationField':i[7],'EmployeeCount':int(i[8]),'EmployeeNumber':int(i[9]),'EnvironmentSatisfaction':int(i[10]),'Gender':i[11],'HourlyRate':int(i[12]),'JobInvolvement':int(i[13]),'JobLevel':int(i[14]),'JobRole':i[15],'JobSatisfaction':int(i[16]),'MaritalStatus':i[17],'MonthlyIncome':int(i[18]),'MonthlyRate':int(i[19]),'NumCompaniesWorked':int(i[20]),'Over18':1,'OverTime':int(i[22]),'PercentSalaryHike':int(i[23]),'PerformanceRating':int(i[24]),'RelationshipSatisfaction':int(i[25]),'StandardHours':int(i[26]),'StockOptionLevel':int(i[27]),'TotalWorkingYears':int(i[28]),'TrainingTimesLastYear':int(i[29]),'WorkLifeBalance':int(i[30]),'YearsAtCompany':int(i[31]),'YearsInCurrentRole':int(i[32]),'YearsSinceLastPromotion':int(i[33]),'YearsWithCurrManager':int(i[34])})
