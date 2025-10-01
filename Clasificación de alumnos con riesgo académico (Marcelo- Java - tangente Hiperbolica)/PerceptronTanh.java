package riesgo;

import java.util.*;

public class PerceptronTanh {
    private double[] W;
    private int nFeatures;
    private Random rnd = new Random(42);

    public PerceptronTanh(int nFeatures) {
        this.nFeatures = nFeatures;
        W = new double[nFeatures + 1]; // +1 bias
        for (int i = 0; i < W.length; i++) {
            W[i] = (rnd.nextDouble() - 0.5) * 0.1;
        }
    }

    private double tanh(double z) { return Math.tanh(z); }

    public double forward(double[] x) {
        double z = 0;
        for (int i = 0; i < nFeatures; i++) z += x[i] * W[i];
        z += W[nFeatures]; // bias
        return tanh(z);
    }

    public void train(double[][] X, int[] y, double lr, int epochs) {
        for (int e = 0; e < epochs; e++) {
            for (int s = 0; s < X.length; s++) {
                double out = forward(X[s]);
                double target = y[s];
                double dLdz = (out - target) * (1 - out * out);
                for (int i = 0; i < nFeatures; i++) {
                    W[i] -= lr * dLdz * X[s][i];
                }
                W[nFeatures] -= lr * dLdz;
            }
        }
    }

    public int predict(double[] x) {
        return forward(x) >= 0.5 ? 1 : 0; // ahora devuelve 0 o 1
    }

    public double rawOutput(double[] x) {
        return forward(x); // valor entre -1 y 1
    }
}

