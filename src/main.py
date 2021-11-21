from spamClassification import spamClassification

def main():
	bot = spamClassification()
	dataPathList = ["./data/YouTube-Spam-Collection-v1/Youtube01-Psy.csv", "./data/YouTube-Spam-Collection-v1/Youtube02-KatyPerry.csv"]
	modelPath = "./my.model"
	bot.training(dataPathList, modelPath)

if __name__ == "__main__":
	main()
