package com.klgraham.minibrain.network;

import com.klgraham.minibrain.neuron.ActivationFunction;
import com.klgraham.minibrain.neuron.Neuron;

/**
 * Created by klogram on 12/28/15.
 */
public class FeedForwardNeuralNetwork implements NeuralNetwork
{
    /**
     * Number of layers in the network. Includes the input and output layers.
     */
    int numberOfLayers;

    /**
     * Number of inputs to the network. This is the number of rows in the
     * input vector of the first layer. This is essentially the size of the input layer.
     */
    int numberOfInputs;

    private Layer[] layers;


    private FeedForwardNeuralNetwork(final int numberOfLayers, final int numberOfInputs)
    {
        this.numberOfLayers = numberOfLayers;
        this.numberOfInputs = numberOfInputs;
        layers = new Layer[numberOfLayers];
    }

    public static FeedForwardNeuralNetwork build(final int numberOfLayers, final int numberOfInputs,
                                          final ActivationFunction[] functions,
                                          int[] neuronsInLayer)
    {
        FeedForwardNeuralNetwork network = new FeedForwardNeuralNetwork(numberOfLayers, numberOfInputs);

        int nIn = neuronsInLayer[0];
        Layer inputLayer = Layer.buildInputLayer(neuronsInLayer[0]);
        network.layers[0] = inputLayer;

        for (int l = 1; l < numberOfLayers; l++)
        {
            Layer layer = Layer.build(neuronsInLayer[l], nIn, functions[l]);
            nIn = neuronsInLayer[l];
            network.layers[l] = layer;
        }

        return network;
    }

	/**
     * Performs a single feedforward pass, computing the activations for all layers
     * except for the input layer.
     * @param inputs
     * @return The activations of the output layer
     */
    @Override
    public double[] predict(double[] inputs)
    {
        double[] layerInputs = inputs;
        for (Layer layer : layers)
        {
            layerInputs = layer.process(layerInputs);
        }
        return layers[numberOfLayers - 1].getOutput();
    }

	/**
     * Train the neural net
     * @param data Training examples. Each row is one training example
     * @param labels Training example labels. Corresponds to each row of data
     */
    @Override
    public void train(final double[][] data, final double[] labels)
    {
        double alpha = 0.1;
        int N = labels.length;

        for (int n = 0; n < N; n++)
        {
            gradientDescent(data[n], labels[n], alpha);
        }
    }

    private void gradientDescent(double[] x, double y, double alpha)
    {
        // feedforward
        double[] a = predict(x);

        // backpropagation for output layer
        Layer outputLayer = layers[numberOfLayers-1];
        double[] deltas = new double[a.length];

        for (int i = 0; i < a.length; i++)
        {
            double yMinusA = y - a[i];
            double df = a[i] * (1. - a[i]);
            deltas[i] = -yMinusA * df;
        }
        outputLayer.setDeltas(deltas);

        for (int l = numberOfLayers - 2; l >= 0; l--)
        {
            Layer layer = layers[l];
            Layer nextLayer = layers[l+1];

            deltas = new double[layer.numberOfNeurons];
            double[] deltasOfNextLayer = nextLayer.getDeltas();

            for (int i = 0; i < layer.numberOfNeurons; i++)
            {
                double sum = 0;
                Neuron n = layer.getNeuron(i).get();
                double aj = n.getOutput();
                double df =  aj * (1. - aj);

                for (int j = 0; j < layers[l+1].numberOfNeurons; j++)
                {
                    double[] w = nextLayer.getNeuron(j).get().weights;
                    sum += w[i] * deltasOfNextLayer[j];
                }
                deltas[i] = sum * df;
            }
            layer.setDeltas(deltas);
        }

        for (int l = 0; l < numberOfLayers-1; l++)
        {
            Layer layer = layers[l];
            Layer nextLayer = layers[l+1];

            double[] aj = layer.getOutput();
            double[] deltasOfNextLayer = nextLayer.getDeltas();

            for (int i = 0; i < nextLayer.numberOfNeurons; i++)
            {
                double d = deltasOfNextLayer[i];
                double[] dJdW = nextLayer.getNeuron(i).get().getdJdW();
                Neuron nNext = layer.getNeuron(i).get();
                for (int j = 0; j < layer.numberOfNeurons; j++)
                {
                    dJdW[j] = aj[j] * d;
                }
                nNext.setdJdW(dJdW);
                nNext.setdJdb(d);
            }
        }

        for (int l = 1; l < numberOfLayers; l++)
        {
            Layer layer = layers[l];

            for (int i = 0; i < layer.numberOfNeurons; i++)
            {
                Neuron neuron = layer.getNeuron(i).get();
                double[] dJdW = neuron.getdJdW();
                double[] w = neuron.weights;
                double dJdb = neuron.getdJdb();

                for (int j = 0; j < dJdW.length; j++)
                {
                    w[j] -= alpha * dJdW[j];
                }
                neuron.bias -= alpha * dJdb;
            }
        }
    }

    public static void main(String[] args)
    {
        ActivationFunction[] activationFunctions = {ActivationFunction.SIGMOID, ActivationFunction.SIGMOID, ActivationFunction.IDENTITY};
        int[] neuronsInLayers = {3, 3, 1};
        FeedForwardNeuralNetwork network = FeedForwardNeuralNetwork.build(3, 3, activationFunctions, neuronsInLayers);
        double[][] data = {{5, 6, 1}, {1, -3, 12}};
        double[] labels = {1, 6};

        // feedforward
//        double[] output = network.predict(inputs);
//        for (double d : output) System.out.println(d);

        network.train(data, labels);
    }
}
