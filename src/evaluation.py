import joblib
from spamClassification import spamClassification
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# load model
model = joblib.load('my.model')
# load data
bot = spamClassification()
dataPathList = ["../data/Youtube04-Eminem.csv"]
comments, label = bot.loadData(dataPathList)
attrL = []
attrU = []
attrH = []
attrC = []
for comment in comments:
	attrL.append(bot.commentLen(comment))
	attrU.append(bot.isUrl(comment))
	attrH.append(bot.findAttrH(comment))
	comment = bot.dealwithComment(comment)
	attrC.append(bot.cos(comment))
testX = []	
for i in range(len(attrL)):
	testX.append([attrL[i], attrU[i], attrH[i], attrC[i]])
testY = label

# evalute
pred_model = model.predict(testX)
print("confusion matrix:")
print(confusion_matrix(testY, pred_model))
print('Accuracy of the model is {:.0%}'.format(accuracy_score(testY, pred_model)))
