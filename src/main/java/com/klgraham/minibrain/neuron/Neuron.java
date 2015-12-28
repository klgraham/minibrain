package com.klgraham.minibrain.neuron;

import Jama.Matrix;

import java.util.function.Function;

/**
 * Created by klogram on 12/27/15.
 */
public class Neuron
{
    public double[] weights;
    public double bias;
    private Function<Double, Double> f;

    public Neuron(double[] w, double bias, ActivationFunction f)
    {
        this.weights = w;
        this.bias = bias;
        this.f = f.get();
    }

    private double z(double[] inputs)
    {
        Matrix x = new Matrix(inputs, 1);
        Matrix w = new Matrix(weights, 1);
        return w.times(x.transpose()).trace() + bias;
    }

    public double output(double[] inputs)
    {
        return f.apply(z(inputs));
    }

    public static void main(String[] args)
    {
        double[] inputs = {1, 0, 1};
        double[] weights = {6, 2, 2};
        double bias = 10;
        Neuron neuron = new Neuron(weights, bias, ActivationFunction.SIGMOID);
        double output = neuron.output(inputs);
        System.out.println(output);
    }
}
