#ifndef CONFIG_H
#define CONFIG_H

struct TaskConfig{
	char  *fileTrainImageMNIST;
	char  *fileTrainLabelsMNIST;
	char  *fileTestImageMNIST;
	char  *fileTestLabelsMNIST;
	int    numberHiddenNeurons;
    int    numberEpochs;
    double learningRate;
    double errorCrossEntropy;
  
	TaskConfig()
	{	
		fileTrainImageMNIST = new char[500];
		fileTrainLabelsMNIST = new char[500];
		fileTestImageMNIST = new char[500];
		fileTestLabelsMNIST = new char[500];
		
		numberHiddenNeurons = 100;
		numberEpochs = 10;
		learningRate = 0.008;
		errorCrossEntropy = 0.005;
	}
} ; 

extern TaskConfig task;

void read_config();

#endif