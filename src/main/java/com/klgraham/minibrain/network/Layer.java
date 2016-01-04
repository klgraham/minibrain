package com.klgraham.minibrain.network;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import com.klgraham.minibrain.neuron.ActivationFunction;
import com.klgraham.minibrain.neuron.Neuron;

/**
 * Represents a layer of neurons in a neural network.
 *
 * Created by klogram on 12/27/15.
 */
public class Layer
{
    /**
     * Description of the layer.
     */
    private String description = "";
    private List<Neuron> neurons;

    /**
     * Number of neurons in the layer.
     */
    int numberOfNeurons;

    /**
     * Number of inputs to the layer.
     */
    int numberOfInputs;

    private ActivationFunction f;
    private double[] output;

    private Layer(final int numberOfNeurons, final int numberOfInputs, ActivationFunction f)
    {
        this.numberOfNeurons = numberOfNeurons;
        this.numberOfInputs = numberOfInputs;
        this.f = f;
        neurons = new ArrayList<>(numberOfNeurons);
    }

    /**
     * Creates a layer of neurons, with weights initialized as ~N(0, \epsilon^2).
     * @param numberOfNeurons Number of Neurons in layer
     * @param numberOfInputs Number of rows in the layer's input (column) vector.
     * @param f Activation function
     * @return Layer of neurons
     */
    public static Layer build(final int numberOfNeurons, final int numberOfInputs, final int numberOfFeatures, ActivationFunction f)
    {
        Layer layer = new Layer(numberOfNeurons, numberOfInputs, f);

        IntStream.rangeClosed(1, numberOfNeurons).forEach(i -> {
            Neuron n = new Neuron(f);
            n.init(numberOfInputs);
            layer.neurons.add(n);
        });
        return layer;
    }

    /**
     * Returns a description of the layer.
     * @return
     */
    public String getDescription() {
        if (!description.isEmpty())
        {
            return description;
        }
        else
        {
            return "Layer{neurons: " + numberOfNeurons +
                    ", activation: " + f.name() +
                    ", inputs: " + numberOfInputs + "}";
        }
    }

    /**
     * Sets the description of the layer.
     * @param description
     */
    public void setDescription(String description) {
        this.description = description;
    }

    /**
     * Computes the output of the entire layer.
     * @param inputs
     * @return
     */
    public double[] process(double[] inputs)
    {
        double[] outputs = new double[numberOfNeurons];
        IntStream.range(0, numberOfNeurons).forEach(i -> {
            Neuron n = neurons.get(i);
            outputs[i] = n.process(inputs);
        });
        this.output = outputs;
        return outputs;
    }

    public double[] getOutput() {
        return output;
    }

    @Override
    public String toString() {
        return getDescription();
    }

    public static void main(String[] args)
    {
        double[] inputs = {1, 0, 1};
        double[] weights = {6, 2, 2};
        double bias = 10;
        Layer layer = Layer.build(4, 3, 1, ActivationFunction.SIGMOID);

        for (Neuron n : layer.neurons)
        {
            n.weights = weights;
            n.bias = bias;
        }
        layer.process(inputs);
        for (double d : layer.getOutput())
        {
            System.out.println(d);
        }
    }
}
