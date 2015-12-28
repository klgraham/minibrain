package com.klgraham.minibrain.neuron;

import java.util.function.Function;

/**
 * Created by klogram on 12/27/15.
 */
public enum ActivationFunction
{
    SIGMOID, STEP;

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
            default:
                f = sigmoid;
                break;
        }
        return f;
    }

    Function<Double, Double> sigmoid = z -> 1.0 / (1.0 + Math.exp(-z));
    Function<Double, Double> step = z -> z > 0 ? 1.0 : 0;
}
