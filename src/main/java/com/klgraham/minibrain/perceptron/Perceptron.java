package com.klgraham.minibrain.perceptron;

import Jama.Matrix;

/**
 * Created by klogram on 12/27/15.
 */
public class Perceptron
{
    // binary inputs x_0, ..., x_{N-1}
    public double[] inputs; // only one row exists

    public int N;

    // weights
    public double[] weights; // only one row exists

    public double threshold;

    public Perceptron(double[] inputs, double[] weights, double threshold) {
        N = inputs.length;
        this.inputs = inputs;
        this.weights = weights;
        this.threshold = threshold;
    }

    public Integer output()
    {
        double sum;
        Matrix x = new Matrix(inputs, 1);
        Matrix w = new Matrix(weights, 1);
        x.print(1,3);
        Matrix xDotX = x.times(w.transpose());
        sum = xDotX.get(0,0) - threshold;
        return sum > 0 ? 1 : 0;
    }

    public static void main(String[] args)
    {
        double[] binaryInputs = {1, 0, 1};
        double[] weights = {6, 2, 2};
        double threshold = 10;
        Perceptron perceptron = new Perceptron(binaryInputs, weights, threshold);
        double output = perceptron.output();
        System.out.println(output);
    }
}
