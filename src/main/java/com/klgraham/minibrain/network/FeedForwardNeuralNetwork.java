package com.klgraham.minibrain.network;

import Jama.Matrix;
import com.klgraham.minibrain.neuron.ActivationFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by klogram on 12/28/15.
 */
public class FeedForwardNeuralNetwork implements NeuralNetwork
{
    /**
     * Number of layers in the network.
     */
    int numberOfLayers;

    /**
     * Number of inputs to the network.
     * This is the number of columns in the input matrix.
     */
    int numberOfInputs;

    /**
     * Number of features per input. This is also the number of rows in the
     * input matrix.
     */
    int numberOfFeatures;

    private List<Layer> layers;

    private FeedForwardNeuralNetwork(final int numberOfLayers, final int numberOfInputs, final int numberOfFeatures)
    {
        this.numberOfLayers = numberOfLayers;
        this.numberOfInputs = numberOfInputs;
        this.numberOfFeatures = numberOfFeatures;
        layers = new ArrayList<>(numberOfLayers);
    }

    public static FeedForwardNeuralNetwork build(final int numberOfLayers, final int numberOfInputs,
                                          final int numberOfFeatures, ActivationFunction[] functions,
                                          int[] neuronsInLayer)
    {
        FeedForwardNeuralNetwork network = new FeedForwardNeuralNetwork(numberOfLayers, numberOfInputs, numberOfFeatures);

        int nIn = numberOfInputs;
        int nFeatures = numberOfFeatures;
        for (int l = 0; l < numberOfLayers; l++)
        {
            Layer layer = Layer.build(neuronsInLayer[l], nIn, nFeatures, functions[l]);
            nIn = neuronsInLayer[l];
            nFeatures = 1;
            network.layers.add(layer);
        }

        return network;
    }

    @Override
    public double[] predict(double[][] inputs, double bias)
    {
        Layer layer0 = layers.get(0);
        double[][] layerInputs = {layer0.process(inputs, bias)};

        for (int l = 1; l < numberOfLayers; l++)
        {
            Layer layer = layers.get(l);
            layerInputs[0] = layer.process(layerInputs, bias);
        }
        return layers.get(numberOfLayers - 1).getOutput();
    }

    @Override
    public void train(double[][] data, double[] labels)
    {

    }

    public static void main(String[] args)
    {
        ActivationFunction[] a = {ActivationFunction.SIGMOID, ActivationFunction.SIGMOID};
        int[] n = {4,2};
        FeedForwardNeuralNetwork network = FeedForwardNeuralNetwork.build(2, 3, 1, a, n);
        double[][] inputs = {{5, 6, 1}};
        double[] output = network.predict(inputs, 1);
        for (double d : output) System.out.println(d);
    }
}
