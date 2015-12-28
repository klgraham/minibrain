package com.klgraham.minibrain.network;

import com.klgraham.minibrain.neuron.ActivationFunction;
import com.klgraham.minibrain.neuron.Neuron;

/**
 * A layer is an array of Neurons
 * Created by klogram on 12/27/15.
 */
public class Layer
{
    Neuron[] neurons;
    double biasUnit = 1.0;
    int numberOfNeurons;
    int numberOfInputs;
    ActivationFunction f;

    private Layer(int numberOfNeurons, int numberOfInputs, ActivationFunction f)
    {
        this.numberOfNeurons = numberOfNeurons;
        this.numberOfInputs = numberOfInputs;
        this.f = f;
    }

    public Layer init(int numberOfNeurons, int numberOfInputs, ActivationFunction f)
    {
        Layer layer = new Layer(numberOfNeurons, numberOfInputs, f);
        for (Neuron n : neurons)
        {
            n = new Neuron(0, f);
            n.initRandom(numberOfInputs);
        }
        return layer;
    }
}
