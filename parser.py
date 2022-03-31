from pathlib import Path

inputfile = open('log.txt')
outputfile = open('out.csv', 'w')

outputfile.writelines("Epoch,LR,Train Loss, Accuracy, Acc_class, mIoU,fwIoU \n")


epoch = 0
lr ="0"
train_loss ="0"
acc = "0"
acc_class = "0"
mIoU = "0"
fwIoU = "0"

for line in inputfile:
	wordList = line.split()
	for word in wordList:
		if word == "learning":
			lr=wordList[5][:-1]
			epoch=epoch+1
		if word == "Loss:":
			train_loss=wordList[1]
		if "Acc:" in word:
			acc =word[4:-1]
		if "Acc_class:" in word:
			acc_class = word[10:-1]
		if "mIoU:" in word:
			mIoU = word[5:-1]
		if word == "fwIoU:":
			fwIoU = wordList[4]
			outputfile.write(str(epoch)+','+lr+','+train_loss+','+acc+','+acc_class+','+mIoU+','+fwIoU+"\n")
	

inputfile.close()
outputfile.close()
