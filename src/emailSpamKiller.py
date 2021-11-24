import numpy as np
import sklearn.utils
from sklearn.model_selection import train_test_split
import glob
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

def loadData(dirPath):
    files = []
    filePath = glob.glob(dirPath + '/*')
    for path in filePath:
        with open(path, 'rb') as f:
            byteContent = f.read()
            strContent = byteContent.decode('utf-8', errors='ignore')
            files.append(strContent)
    return files


def emailLen(email):
		return len(email)

def isUrl(comment):
    #if "http" in comment or "https" in comment:
    if "https" in comment:
        return 1
    else:
        return 0

def main():
    #load data
    print("Loading data")
    spam = loadData("../data/spam_2")
    easy_ham = loadData("../data/easy_ham")
    hard_ham = loadData("../data/hard_ham")
    emails = spam + easy_ham + hard_ham
    label = np.concatenate((np.ones(len(spam), dtype=int), np.zeros(len(easy_ham) + len(hard_ham), dtype=int)))
    emails, label = sklearn.utils.shuffle(emails, label, random_state=42)
    print("preparing data")
    attrU = []
    attrL = []
    for email in emails:
        attrU.append(isUrl(email))
        attrL.append(emailLen(email))
    x = []	
    for i in range(len(attrU)):
        x.append([attrU[i], attrL[i]])
    y = label.tolist()
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    model = dtc(criterion = 'entropy', max_depth = 4)
    model.fit(x_train, y_train)
    pred_model = model.predict(x_test)
    print("confusion matrix:")
    print(confusion_matrix(y_test, pred_model))
    print('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model)))

if __name__ == "__main__":
	main()
