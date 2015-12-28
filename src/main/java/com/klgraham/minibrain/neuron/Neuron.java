package com.klgraham.minibrain.neuron;

import Jama.Matrix;

import java.util.Random;
import java.util.function.Function;

/**
 * A Neuron has N inputs, each of which is a vector of size K, and produces a single output value,
 * as determined by the activation function. For each input, there is a corresponding vector of weights.
 *
 * Created by klogram on 12/27/15.
 */
public class Neuron
{
    /**
     * K x N matrix of weights. Each column contains the weight vector for the corresponding input vector.
     */
    public double[][] weights;

    public double bias;
    private double output;

    /**
     * Activation function h_{w,b}(x),
     * where x = (x_0, ..., x_{N-1}), each x_i is a column matrix
     */
    private Function<Double, Double> f;


    public Neuron(double[][] w, final double bias, ActivationFunction f)
    {
        this.weights = w;
        this.bias = bias;
        this.f = f.get();
    }

    public Neuron(final double bias, ActivationFunction f)
    {
        this.bias = bias;
        this.f = f.get();
        this.weights = null;
    }

    private double z(final double[][] inputs)
    {
        Matrix x = new Matrix(inputs);
        Matrix w = new Matrix(weights);
        return w.transpose().times(x).trace() + bias;
    }

    /**
     * Computes the output of the Neuron.
     * @param inputs
     * @return Neuron's output value.
     */
    public double process(final double[][] inputs)
    {
        output = f.apply(z(inputs));
        return output;
    }

    public static void main(String[] args)
    {
        double[][] inputs = {{1, 0, 1}};
        double[][] weights = {{6, 2, 2}};
        double bias = 10;
        Neuron neuron = new Neuron(weights, bias, ActivationFunction.SIGMOID);
        neuron.process(inputs);
        System.out.println(neuron.getOutput());
    }

    public double getOutput() {
        return output;
    }

    public void init(final int numInputs, final int numFeatures)
    {
        this.weights = new double[numFeatures][numInputs];
        for (int i = 0; i < numInputs; i++) {
            for (int j = 0; j < numFeatures; j++) {
                weights[i][j] = 0.0;
            }
        }
    }

    private Random r = new Random();

    public void initRandom(final int numInputs, final int numFeatures)
    {
        this.weights = new double[numFeatures][numInputs];
        for (int i = 0; i < numInputs; i++) {
            for (int j = 0; j < numFeatures; j++) {
                weights[i][j] = r.nextDouble();
            }
        }
    }

    public void setActivationFunction(ActivationFunction f) {
        this.f = f.get();
    }
}
