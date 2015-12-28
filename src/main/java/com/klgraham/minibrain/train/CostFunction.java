package com.klgraham.minibrain.train;

import com.klgraham.minibrain.neuron.ActivationFunction;

import java.util.function.Function;

/**
 * Created by klogram on 12/28/15.
 */
public class CostFunction
{
    /**
     * Number of training examples.
     * This is the number of columns in the input matrix x.
     */
    int numberOfExamples;

    /**
     * Number of features per example. This is also the number of rows in the
     * input matrix x.
     */
    int numberOfFeatures;

    double[][] weights;
    double weightDecayParameter;
    Function<Double, Double> f;

    public CostFunction(int numberOfExamples, int numberOfFeatures, double[][] weights, ActivationFunction f, double weightDecayParameter)
    {
        this.numberOfExamples = numberOfExamples;
        this.numberOfFeatures = numberOfFeatures;
        this.weights = weights;
        this.f = f.get();
        this.weightDecayParameter = weightDecayParameter;
    }

    public double compute(double[][] x, double[] y, double bias)
    {
        double costSum = 0;
        for (int i = 0; i < numberOfExamples; i++)
        {
            double yy = y[i];
            for (int j = 0; j < numberOfFeatures; j++)
            {
                double jSum = f.apply(weights[j][i] * x[j][i] + bias) - yy;
                jSum *= 0.5 * jSum;
                costSum += jSum;
            }
        }

        return costSum / (double)numberOfExamples;
    }
}
