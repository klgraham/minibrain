package com.klgraham.minibrain.neuron;

import java.util.function.Function;

/**
 * Created by klogram on 12/27/15.
 */
public enum ActivationFunction
{
    SIGMOID(0), STEP(1);

    private final int functionCode;

    ActivationFunction(int functionCode)
    {
        this.functionCode = functionCode;
    }

    public Function<Double, Double> get()
    {
        switch (functionCode)
        {
            case 0: return Functions.sigmoid;
            case 1: return Functions.step;
            default: return Functions.sigmoid;
        }
    }

    Function<Double, Double> sigmoid = z -> 1.0 / (1.0 + Math.exp(-z));
    Function<Double, Double> step = z -> z > 0 ? 1.0 : 0;
}
