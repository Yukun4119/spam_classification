import numpy as np
import glob
def loadData(dirPath):
    None


def main():
#	f = open("../data/spam_2/01400.b444b69845db2fa0a4693ca04e6ac5c5")
	f = open("../data/hard_ham/00250.c7603b27a45284d12b49adf767b2b6fa")
	lines = f.readlines()
#lines = lines[lines.index('\n\n'):]
	for line in lines:
		print(line)

	f.close()


if __name__ == "__main__":
	main()
