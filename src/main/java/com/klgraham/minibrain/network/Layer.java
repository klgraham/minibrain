package com.klgraham.minibrain.network;

import com.klgraham.minibrain.neuron.ActivationFunction;
import com.klgraham.minibrain.neuron.Neuron;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

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
    List<Neuron> neurons;

    /**
     * Number of neurons in the layer.
     */
    int numberOfNeurons;

    /**
     * Number of inputs to the layer, and to each neuron in the layer.
     * This is the number of columns in the input matrix.
     */
    int numberOfInputs;

    /**
     * Number of features per input. This is also the number of rows in the
     * input matrix.
     */
    int numberOfFeatures;

    ActivationFunction f;

    private Layer(final int numberOfNeurons, final int numberOfInputs, final int numberOfFeatures, ActivationFunction f)
    {
        this.numberOfNeurons = numberOfNeurons;
        this.numberOfInputs = numberOfInputs;
        this.numberOfFeatures = numberOfFeatures;
        this.f = f;
        neurons = new ArrayList<>(numberOfNeurons);
    }

    /**
     * Creates a layer of neurons, with weights initialized on [0, 1].
     * @param numberOfNeurons
     * @param numberOfInputs
     * @param numberOfFeatures
     * @param f
     * @return Layer of neurons
     */
    public Layer init(final int numberOfNeurons, final int numberOfInputs, final int numberOfFeatures, ActivationFunction f)
    {
        Layer layer = new Layer(numberOfNeurons, numberOfInputs, numberOfFeatures, f);

        IntStream.range(1, numberOfNeurons).forEach(i -> {
            Neuron n = new Neuron(f);
            n.initRandom(numberOfInputs, numberOfFeatures);
            neurons.add(n);
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
                    ", features: " + numberOfFeatures +
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
    public double[] process(double[][] inputs, final double bias)
    {
        double[] outputs = new double[numberOfNeurons];
        IntStream.range(0, numberOfNeurons-1).forEach(i -> {
            Neuron n = neurons.get(i);
            outputs[i] = n.process(inputs, bias);
        });
        return outputs;
    }

    @Override
    public String toString() {
        return getDescription();
    }
}
