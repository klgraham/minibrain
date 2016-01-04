package com.klgraham.minibrain.network;

import java.util.Optional;
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
    private Neuron[] neurons;

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
    private double[] deltas;
    private double[] zs;
    private boolean isInputLayer;

    private Layer(final int numberOfNeurons, final int numberOfInputs, ActivationFunction f)
    {
        this.numberOfNeurons = numberOfNeurons;
        this.numberOfInputs = numberOfInputs;
        this.f = f;
        neurons = new Neuron[numberOfNeurons];
        deltas = new double[numberOfNeurons];
        zs = new double[numberOfNeurons];
        isInputLayer = false;
    }

    /**
     * Creates a layer of neurons, with weights initialized as ~N(0, \epsilon^2).
     * @param numberOfNeurons Number of Neurons in layer
     * @param numberOfInputs Number of rows in the layer's input (column) vector.
     * @param f Activation function
     * @return Layer of neurons
     */
    public static Layer build(final int numberOfNeurons, final int numberOfInputs, ActivationFunction f)
    {
        Layer layer = new Layer(numberOfNeurons, numberOfInputs, f);

        IntStream.rangeClosed(1, numberOfNeurons).forEach(i -> {
            Neuron n = new Neuron(f);
            n.init(numberOfInputs);
            layer.neurons[i-1] = n;
        });
        return layer;
    }

    /**
     * Creates an input layer
     * @param numberOfUnits Number of units in layer
     * @return
     */
    public static Layer buildInputLayer(final int numberOfUnits)
    {
        Layer layer = new Layer(numberOfUnits, 1, ActivationFunction.IDENTITY);
        layer.isInputLayer = true;

        IntStream.rangeClosed(1, numberOfUnits).forEach(i -> {
            Neuron n = new Neuron(ActivationFunction.IDENTITY);
            n.init(numberOfUnits);
            layer.neurons[i-1] = n;
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

        if (!isInputLayer)
        {
            IntStream.range(0, numberOfNeurons).forEach(i -> {
                Neuron n = neurons[i];
                outputs[i] = n.process(inputs);
                zs[i] = n.getZ();
            });
            this.output = outputs;
        }
        else
        {
            setOutput(inputs);
        }
        return outputs;
    }

    public double[] getOutput() {
        return output;
    }

    public void setOutput(final double[] output)
    {
        if (isInputLayer)
        {
            this.output = output;
        }
    }

    public double[] getDeltas()
    {
        return deltas;
    }

    public void setDeltas(final double[] deltas)
    {
        this.deltas = deltas;
    }

    public double[] getZs()
    {
        return zs;
    }

    public Optional<Neuron> getNeuron(int i)
    {
        if (isValidNeuronIndex(i))
        {
            return Optional.of(neurons[i]);
        }
        else
        {
            return Optional.empty();
        }
    }

    private boolean isValidNeuronIndex(int i)
    {
        return i >= 0 && i < numberOfNeurons;
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
        Layer layer = Layer.build(4, 3, ActivationFunction.SIGMOID);

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
