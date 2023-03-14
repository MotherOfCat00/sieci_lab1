import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing

# treningowa
train = pd.read_table(r'D:\Studia\drugsComTrain_raw.tsv')
# Enkoder
label_encoder = preprocessing.LabelEncoder()
train['drugName'] = label_encoder.fit_transform(train['drugName'])
print(train.describe())
# train.hist()
# train.plot(kind='box', sharex=False, sharey=False)
# scatter_matrix(train)

# walidacyjna
validation = pd.read_table(r'D:\Studia\drugsComTest_raw.tsv')
# Enkoder
label_encoder = preprocessing.LabelEncoder()
validation['drugName'] = label_encoder.fit_transform(validation['drugName'])
print(validation.describe())
# validation.hist()
# validation.plot(kind='box', sharex=False, sharey=False)
# scatter_matrix(validation)

# Split the data
Y_train = train['drugName'].copy()
Y_validation = validation['drugName'].copy()
X_train = train[['Unnamed: 0', 'rating', 'usefulCount']].copy()
X_validation = validation[['Unnamed: 0', 'rating', 'usefulCount']].copy()

Y_train = Y_train.values
Y_validation = Y_validation.values
X_train = X_train.values
X_validation = X_validation.values

scoring = 'accuracy'
seed = 7

# Spot Check Algorithms
models = [
    ('LR', LogisticRegression(solver='sag')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

plt.show()
