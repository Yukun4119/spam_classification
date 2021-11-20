import numpy as np
import glob
def loadData(dirPath):
	files = []
    filePath = glob.glob(dirPath + '/*')
    for path in filePath:
        with open(path, 'rb') as f:
            byteContent = f.read()
            strContent = byte_content.decode('utf-8', errors='ignore')
            files.append(str_content)
    return files

def main():
	spam = load_dataset("../data/spam_2")
	hard_ham = load_dataset("../data/hard_ham")

if "__name__" == __main__:
	main()
