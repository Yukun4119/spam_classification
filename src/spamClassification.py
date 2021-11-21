import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re 

class spamClassification:
	def __init__(self):
		None

	def commentLen(self, comment):
		return len(comment)

	def isUrl(self, comment):
		if "http" in comment or "https" in comment:
			return 1
		else:
			return 0

	def findAttrH(self, s):
		andFlag = 0
		for cha in s:
			if cha == '&':
				andFlag = 1
			if cha == ';':
				if andFlag == 1:
					return 1
				else:
					return 0
		return 0

	def wordVector(self, splitedStr):
		wordDic = {"check": 0, "video":1, "song":2, "youtube":3, "like":4, "subscribe": 5, "please": 6, "love": 7, "channel":8, "music": 9} 
		curVector = [0] * 10
		for word in splitedStr:
			if word in wordDic.keys():
				curVector[wordDic[word]] += 1
		return curVector
		

	def cos(self, splitedStr):
		v = self.wordVector(splitedStr)
		x = np.array(v)
		defaultVector = [472, 300, 270, 244, 218, 206, 192, 189, 182, 114]
		y = np.array(defaultVector)
		num = x.dot(y.T)
		if num == 0:
			return 0
		denom = np.linalg.norm(x) * np.linalg.norm(y)
		return num / denom


	def training(self, dataPathList, modelPath):
		print("Loading data")
		comments, label = self.loadData(dataPathList)
		print("preparing data")
		attrL = []
		attrU = []
		attrH = []
		attrC = []
		for comment in comments:
			attrL.append(self.commentLen(comment))
			attrU.append(self.isUrl(comment))
			attrH.append(self.findAttrH(comment))
			comment = self.dealwithComment(comment)
			attrC.append(self.cos(comment))
		x = []	
		for i in range(len(attrL)):
			x.append([attrL[i],attrU[i], attrH[i], attrC[i]])
		y = label

		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=40)
		model = dtc(criterion = 'entropy', max_depth = 4)
		model.fit(x_train, y_train)
		pred_model = model.predict(x_test)
		print("confusion matrix:")
		print(confusion_matrix(y_test, pred_model))
		print('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model)))
		joblib.dump(model, modelPath)
		print("Finish training")
		print("model is in", modelPath)

	def loadData(self, dataPathList):
		dfList = []
		for dataPath in dataPathList:
			df = pd.read_csv(dataPath)
			dfList.append(df)
		df = pd.concat(dfList)
		comment = df['CONTENT'].values
		label = df['CLASS'].values
		return comment, label

	def dealwithComment(self, s):
		res = s.split(" ")
		for i in range(len(res)):
			ts = "".join(re.findall("[A-Za-z]+",res[i].lower()))
			if ts is not None:
				res[i] = ts
		return res

def main():
	bot = spamClassification()
	dataPathList = ["./data/YouTube-Spam-Collection-v1/Youtube01-Psy.csv", "./data/YouTube-Spam-Collection-v1/Youtube02-KatyPerry.csv"]
	modelPath = "./my.model"
	bot.training(dataPathList, modelPath)

if __name__ == "__main__":
	main()
