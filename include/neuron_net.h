#ifndef NEURON_NET_H
#define NEURON_NET_H

class NeuronNet {

private:

	int numberInputNeurons;
	int numberHiddenNeurons;
	int numberOutputNeurons;

	double *inputX;
	double *outputY;
	double *hiddenOutputs;
	double *outputsNet;
	double **weightsLayer1;
	double **weightsLayer2;

	void initializeWeights();
	void calculateHiddenOutputs();
	void calculateOutputs();
	void calculateGradientErrorFunction(double **gradientWeightsLayer1, double **gradientWeightsLayer2);
	void correctWeights(double **gradientWeightsLayer1, double **gradientWeightsLayer2, double learningRate);
	void backPropagation(double learningRate);
	double calculateValueErrorFunction(double **trainData, double *trainLabel, int numberTrainImage);
	void setRandomOrder(int *order, int size);

public:

	NeuronNet(int _numberInputNeurons, int _numberHiddenNeurons, int _numberOutputNeurons);	
	~NeuronNet();

	double calculatePrecision(double **data, double *label, int numberImage);
	void trainNeuronNetwork(double **trainData, double *trainLabel, int numberTrainImage, int numberEpochs, double learningRate, double errorCrossEntropy);
};

#endif