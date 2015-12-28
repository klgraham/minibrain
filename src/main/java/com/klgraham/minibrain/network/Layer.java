package com.klgraham.minibrain.network;

import com.klgraham.minibrain.neuron.ActivationFunction;
import com.klgraham.minibrain.neuron.Neuron;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * A layer is an array of Neurons
 * Created by klogram on 12/27/15.
 */
public class Layer
{
    List<Neuron> neurons;
    double biasUnit = 1.0;
    int numberOfNeurons;
    int numberOfInputs;
    ActivationFunction f;

    private Layer(int numberOfNeurons, int numberOfInputs, ActivationFunction f)
    {
        this.numberOfNeurons = numberOfNeurons;
        this.numberOfInputs = numberOfInputs;
        this.f = f;
        neurons = new ArrayList<>(numberOfNeurons);
    }

    public Layer init(int numberOfNeurons, int numberOfInputs, ActivationFunction f)
    {
        Layer layer = new Layer(numberOfNeurons, numberOfInputs, f);
        IntStream.range(1, numberOfNeurons).forEach(i -> {
            Neuron n = new Neuron(0, f);
            n.initRandom(numberOfInputs);
            neurons.add(n);
        });
        return layer;
    }

    public double[] process(double[] inputs)
    {
        double[] outputs = new double[numberOfNeurons];
        IntStream.range(0, numberOfNeurons-1).forEach(i -> {
            Neuron n = neurons.get(i);
            outputs[i] = n.process(inputs);
        });
        return outputs;
    }
}
