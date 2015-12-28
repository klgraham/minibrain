package com.klgraham.minibrain.neuron;

import Jama.Matrix;

import java.util.Random;
import java.util.function.Function;

/**
 * Created by klogram on 12/27/15.
 */
public class Neuron
{
    public double[] weights;
    public double bias;
    private double output;
    private Function<Double, Double> f;

    public Neuron(double[] w, double bias, ActivationFunction f)
    {
        this.weights = w;
        this.bias = bias;
        this.f = f.get();
    }

    public Neuron(double bias, ActivationFunction f)
    {
        this.bias = bias;
        this.f = f.get();
        this.weights = null;
    }

    private double z(double[] inputs)
    {
        Matrix x = new Matrix(inputs, 1);
        Matrix w = new Matrix(weights, 1);
        return w.times(x.transpose()).trace() + bias;
    }

    public double process(double[] inputs)
    {
        output = f.apply(z(inputs));
        return output;
    }

    public static void main(String[] args)
    {
        double[] inputs = {1, 0, 1};
        double[] weights = {6, 2, 2};
        double bias = 10;
        Neuron neuron = new Neuron(weights, bias, ActivationFunction.SIGMOID);
        neuron.process(inputs);
        System.out.println(neuron.getOutput());
    }

    public double getOutput() {
        return output;
    }

    public void init(int n)
    {
        this.weights = new double[n];
        for (int i = 0; i < n; i++)
        {
            weights[i] = 0.0;
        }
    }

    private Random r = new Random();

    public void initRandom(int n)
    {
        this.weights = new double[n];
        for (int i = 0; i < n; i++)
        {
            weights[i] = r.nextDouble();
        }
    }
}
