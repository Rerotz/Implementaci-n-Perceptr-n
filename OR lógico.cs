using System;

class Perceptron {
    double[] w;
    double b;
    int nInputs;
    double lr;

    public Perceptron(int nInputs, double lr = 0.1) {
        this.nInputs = nInputs;
        this.lr = lr;
        var rnd = new Random(42);
        w = new double[nInputs];
        for (int i = 0; i < nInputs; i++)
            w[i] = rnd.NextDouble() * 2 - 1;
        b = rnd.NextDouble() * 2 - 1;
    }

    double Net(double[] x) {
        double s = b;
        for (int i = 0; i < nInputs; i++)
            s += w[i] * x[i];
        return s;
    }

    double Step(double x) => x >= 0 ? 1.0 : 0.0;

    public void Train(double[][] X, int[] Y, int epochs) {
        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < X.Length; i++) {
                double s = Net(X[i]);
                int yhat = (int)Step(s);
                int err = Y[i] - yhat;
                if (err != 0) {
                    for (int j = 0; j < nInputs; j++)
                        w[j] += lr * err * X[i][j];