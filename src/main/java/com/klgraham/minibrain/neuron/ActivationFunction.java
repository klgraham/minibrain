package com.klgraham.minibrain.neuron;

import java.util.function.Function;

/**
 * Created by klogram on 12/27/15.
 */
public enum ActivationFunction
{
    SIGMOID, TANH, STEP, RECTIFIED_LINEAR;

    public Function<Double, Double> get()
    {
        Function<Double, Double> f;
        switch (this)
        {
            case SIGMOID:
                f = sigmoid;
                break;
            case STEP:
                f = step;
                break;
            case TANH:
                f = tanh;
                break;
            case RECTIFIED_LINEAR:
                f = rectified;
                break;
            default:
                f = sigmoid;
                break;
        }
        return f;
    }

    // activation functions
    Function<Double, Double> sigmoid = z -> 1.0 / (1.0 + Math.exp(-z));
    Function<Double, Double> step = z -> z > 0 ? 1.0 : 0;
    Function<Double, Double> tanh = z -> (Math.exp(z) - Math.exp(-z)) / (Math.exp(z) + Math.exp(-z));
    Function<Double, Double> rectified = z -> Math.max(0, z);
}
