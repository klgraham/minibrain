package com.klgraham.minibrain.neuron;

import Jama.Matrix;

import java.util.function.Function;

/**
 * Created by klogram on 12/27/15.
 */
public class Neuron
{
    public double[] x;
    public double[] w;
    public double bias;
    public int numberOfInputs;

    private Function<Double, Double> f;

    public Neuron(double[] x, double[] w, double bias, Function<Double, Double> f) {
        this.x = x;
        this.w = w;
        this.bias = bias;
        this.numberOfInputs = x.length;
        this.f = f;
    }

    private double z()
    {
        Matrix xx = new Matrix(x, 1);
        Matrix ww = new Matrix(w, 1);
        return ww.times(xx.transpose()).trace() + bias;
    }

    public double output()
    {
        return f.apply(z());
    }

    public static void main(String[] args)
    {
        double[] inputs = {1, 0, 1};
        double[] weights = {6, 2, 2};
        double bias = 10;
        Neuron neuron = new Neuron(inputs, weights, bias, Functions.sigmoid);
        double output = neuron.output();
        System.out.println(output);
    }
}
