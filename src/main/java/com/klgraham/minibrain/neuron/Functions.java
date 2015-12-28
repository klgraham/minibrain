package com.klgraham.minibrain.neuron;

import java.util.function.Function;

/**
 * Created by klogram on 12/27/15.
 */
public class Functions
{
    static Function<Double, Double> sigmoid = z -> 1.0 / (1.0 + Math.exp(-z));
    static Function<Double, Double> step = z -> z > 0 ? 1.0 : 0;

}
