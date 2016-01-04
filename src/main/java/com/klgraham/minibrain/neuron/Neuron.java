package com.klgraham.minibrain.neuron;

import java.util.Random;
import java.util.function.Function;

import Jama.Matrix;

/**
 *
 * Created by klogram on 12/27/15.
 */
public class Neuron
{
    /**
     * Column vector of N rows, representing weights between each input and this Neuron
     */
    public double[] weights;

    /**
     * Bias of neuron
     */
    public double bias;

	/**
     * Output of Neuron
     */
    private double output;

    private double z;
    private double[] dJdW;
    private double dJdb;

    /**
     * Activation function h_{w,b}(x),
     * where x = (x_0, ..., x_{N-1})
     */
    private Function<Double, Double> f;

    public Neuron(double[] w, double bias, ActivationFunction f)
    {
        this.weights = w;
        this.bias = bias;
        this.f = f.get();
        this.z = 0;
        this.dJdW = new double[w.length];
        this.dJdb = 0;
    }

    public Neuron(ActivationFunction f)
    {
        this.f = f.get();
        this.weights = null;
        this.bias = 1.0;
        this.dJdW = null;
        this.dJdb = 0;
    }

	/**
     * Computes input to activation function
     * @param inputs Column vector of N rows
     * @return
     */
    private double z(final double[] inputs)
    {
        Matrix x = new Matrix(inputs, inputs.length);
        Matrix w = new Matrix(weights, weights.length);
        return w.transpose().times(x).trace() + bias;
    }

    /**
     * Computes the output of the Neuron.
     * @param inputs Column vector of N rows.
     * @return Neuron's output value.
     */
    public double process(final double[] inputs)
    {
        this.z = z(inputs);
        output = f.apply(this.z);
        return output;
    }

//    public double process(final double[] inputs, final boolean isInputNode)
//    {
//        return inputs;
//    }

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

    public double getZ()
    {
        return z;
    }

    public double[] getdJdW()
    {
        return dJdW;
    }

    public void setdJdW(final double[] dJdW)
    {
        this.dJdW = dJdW;
    }

    public double getdJdb()
    {
        return dJdb;
    }

    public void setdJdb(final double dJdb)
    {
        this.dJdb = dJdb;
    }

    public void init(final int numFeatures)
    {
        this.weights = new double[numFeatures];
        this.dJdW = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            weights[i] = r.nextGaussian();
        }
        bias = epsilon * r.nextGaussian();
    }

    private Random r = new Random();
    private double epsilon = 1.0e-4;

    public void setActivationFunction(ActivationFunction f) {
        this.f = f.get();
    }
}
