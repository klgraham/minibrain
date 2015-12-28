package com.klgraham.minibrain.network;

import Jama.Matrix;
import com.klgraham.minibrain.neuron.ActivationFunction;
import com.klgraham.minibrain.neuron.Neuron;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
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
    private List<Neuron> neurons;

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

    private double bias;
    private Random r = new Random();
    private double epsilon = 1.0e-4;

    ActivationFunction f;
    private double[] output;

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
     * @param f Activatin function
     * @return Layer of neurons
     */
    public static Layer build(final int numberOfNeurons, final int numberOfInputs, final int numberOfFeatures, ActivationFunction f)
    {
        Layer layer = new Layer(numberOfNeurons, numberOfInputs, numberOfFeatures, f);

        IntStream.rangeClosed(1, numberOfNeurons).forEach(i -> {
            Neuron n = new Neuron(f);
            n.init(numberOfInputs, numberOfFeatures);
            layer.neurons.add(n);
            layer.bias = layer.epsilon * layer.r.nextGaussian();
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
    public double[] process(double[][] inputs)
    {
        double[] outputs = new double[numberOfNeurons];
        IntStream.range(0, numberOfNeurons).forEach(i -> {
            Neuron n = neurons.get(i);
            outputs[i] = n.process(inputs, bias);
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
        double[][] inputs = {{1, 0, 1}};
        double[][] weights = {{6, 2, 2}};
        double bias = 10;
        Layer layer = Layer.build(4, 3, 1, ActivationFunction.SIGMOID);

        for (Neuron n : layer.neurons)
        {
            n.weights = weights;
        }
        layer.process(inputs);
        for (double d : layer.getOutput())
        {
            System.out.println(d);
        }
    }
}
