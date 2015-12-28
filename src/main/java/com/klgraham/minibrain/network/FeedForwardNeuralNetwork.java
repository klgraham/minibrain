package com.klgraham.minibrain.network;

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
        for (int l = 0; l < numberOfLayers; l++)
        {
            Layer layer = Layer.build(neuronsInLayer[l], nIn, numberOfFeatures, functions[l]);
            nIn = neuronsInLayer[l];
            network.layers.add(layer);
        }

        return network;
    }

    @Override
    public double[] predict(double[][] inputs, double bias)
    {

        return new double[0];
    }

    @Override
    public void train(double[][] data, double[] labels)
    {

    }
}
