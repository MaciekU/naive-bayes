import pandas
import numpy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, metrics
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import model_selection, datasets
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split


kfold_n=9
def Gauss_Multi_NB(kfold_n):
    #for kfold_n in range(2,14):
     #   print(kfold_n)
    (trenowanie_wartosci, testowanie_wartosci, trenowanie_klasy, testowanie_klasy) = \
		train_test_split(wartosci, klasy, test_size=0.4, random_state=1)
    gnb = GaussianNB()
    model=gnb.fit(trenowanie_wartosci, trenowanie_klasy)
    predictions=gnb.predict(testowanie_wartosci)
    #print("Gauss")
    #print(predictions)
    #print(accuracy_score(testowanie_klasy, predictions))
    
    gnb = GaussianNB()
    # jak rozmiar wplywa na dokladność accuracy
    kfold = StratifiedKFold(n_splits=kfold_n)
    # StratifiedKFold(n_splits=10)
    print("F1_mean:")
    results = cross_val_score(gnb, wartosci, klasy, cv=kfold, scoring="f1_micro")
    result = results.mean()
    std = results.std() #standard deviation 
    print(result,std)
    #print(results) #wyniki prób Fold

    gnb = MultinomialNB(alpha=1.0)
    model=gnb.fit(trenowanie_wartosci, trenowanie_klasy)
    predictions=gnb.predict(testowanie_wartosci)
    #print("Multi")
    #print(predictions)
    #print(accuracy_score(testowanie_klasy, predictions))

    
    #print(metrics.classification_report(testowanie_klasy, predictions))
    #print(metrics.confusion_matrix(testowanie_klasy, predictions))

#######################
dane_glass = pandas.read_csv('glass.data', header=None, delimiter=',', engine='python')


dane_glass.columns = ['Id number','RI', 'Na', 'Mg', 'Al', 'Si', 'K',
	'Ca', 'Ba', 'Fe', 'Type of glass' ]
dane_glass.drop('Id number',axis=1, inplace=True)


#dane_glass.hist()
#sns.heatmap(dane_glass.corr(), cmap='BuGn')
#plt.show()

#print(dane_glass.shape)
#print(dane_glass.describe())


dane_wine = pandas.read_csv('wine.data', header=None, delimiter=',', engine='python')


dane_wine.columns = ['Class','Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash',
	'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
	'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']


#print(dane_wine.shape)  # liczba wierszy i kolumn z danymi
#print(dane_wine.describe())

dane_diabetes = pandas.read_csv('pima_indians_diabetes.txt', header=None, delimiter=',', engine='python')


dane_diabetes.columns = ['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick',
	'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class']


#print(dane_diabetes.shape)
#print(dane_diabetes.describe())#do sprawdzenia | występuje kilka błędnych wartości zerowych (plasma_glucose,blood_pres,skin_thick,serum_insu,BMI) chyba że pacjenci są martwi ewentualnie wyparowali patrząc po BMI



#########################
'''
print((dane_diabetes[[0,1,2,3,4,5,6,7,8]] == 0).sum()) # ile jest wartości zerowych w kolumnach danych
print("\n")
dane_diabetes[[1,2,3,4,5]] = dane_diabetes[[1,2,3,4,5]].replace(0, numpy.NaN)#zamień wartości zerowe w podanych kolumnach na NaN
print(dane_diabetes.isnull().sum())  # które i ile jest null
dane_diabetes.dropna(inplace=True)  # usunięcie wierszy z NaN
'''
##########################

#for b in range(0,11):
    #print(b)
#dane_glass['RI'] = pandas.cut(dane_glass['RI'], 12, retbins=False, labels=False) #podzial na zbiory

#dane_glass['RI'] = pandas.qcut(dane_glass['RI'], [0, .25, .5, .75, 1], retbins=False, labels=False) #podzial na kwartyle

#dane_glass['RI'] = numpy.around(dane_glass['RI'], decimals = 2) #zmniejszenie precyzji

wartosci = dane_glass[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']].values


#wartosci = preprocessing.normalize(wartosci)
klasy = dane_glass['Type of glass'].values


print("Glass")
Gauss_Multi_NB(kfold_n)

wartosci = dane_wine[['Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
	'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
	'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']].values
#
#wartosci = preprocessing.normalize(wartosci)
klasy = dane_wine['Class'].values


print("Wine")
Gauss_Multi_NB(kfold_n)


#dane_diabetes[[1,2,3,4,5]] = dane_diabetes[[1,2,3,4,5]].replace(0, numpy.NaN)#zamień wartości zerowe w podanych kolumnach na NaN
#dane_diabetes.fillna(dane_diabetes.mean(), inplace=True)#usunięcie wierszy z NaN

#dane_diabetes[[1,2,3,4,5]] = dane_diabetes[[1,2,3,4,5]].replace(0, numpy.NaN)#zamień wartości zerowe w podanych kolumnach na NaN
#print(dane_diabetes.isnull().sum()) #które i ile jest null
#dane_diabetes.dropna(inplace=True)#usunięcie wierszy z NaN

#dane_diabetes=dane_diabetes.interpolate()

#dane_diabetes[['Plasma_glucose', 'Blood_pres', 'Skin_thick', 'Serum_insu', 'BMI']] = dane_diabetes[['Plasma_glucose', 'Blood_pres', 'Skin_thick', 'Serum_insu', 'BMI']].replace(0, numpy.NaN)#zamień wartości zerowe w podanych kolumnach na NaN


#dane_diabetes=dane_diabetes.interpolate().ffill().bfill() #fill forward | fill backward

#print(dane_diabetes)


wartosci = dane_diabetes[['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick',
	'Serum_insu', 'BMI', 'Diabetes_func', 'Age']].values
#
#wartosci = preprocessing.normalize(wartosci)

klasy = dane_diabetes['Class'].values

print("Diabetes")
Gauss_Multi_NB(kfold_n)




