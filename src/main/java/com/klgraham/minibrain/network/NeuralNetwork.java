package com.klgraham.minibrain.network;

/**
 * Created by klogram on 12/28/15.
 */
public interface NeuralNetwork
{
    double[] predict(final double[] inputs);
    void train(final double[][] data, final double[] labels);
}
