import Foundation

// Función de activación lineal
func linear(_ x: Double) -> Double {
    return x
}

class Perceptron {
    var w: [Double]
    var b: Double
    var lr: Double
    var nInputs: Int

    init(nInputs: Int, lr: Double = 0.1) {
        self.nInputs = nInputs
        self.lr = lr
        self.w = (0..<nInputs).map { _ in Double.random(in: -1...1) }
        self.b = Double.random(in: -1...1)
    }

    func net(_ x: [Double]) -> Double {
        var sum = b
        for i in 0..<nInputs {
            sum += w[i] * x[i]
        }
        return sum
    }

    func predict(_ x: [Double]) -> Double {
        return linear(net(x))
    }

    func train(X: [[Double]], Y: [Double], epochs: Int) {
        for _ in 0..<epochs {
            var dw = Array(repeating: 0.0, count: nInputs)
            var db = 0.0
            for (xi, yi) in zip(X, Y) {
                let s = net(xi)
                let yhat = linear(s)
                let err = yhat - yi
                for j in 0..<nInputs {