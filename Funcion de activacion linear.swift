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
                    dw[j] += err * xi[j]
                }
                db += err
            }
            for j in 0..<nInputs {
                w[j] -= lr * dw[j] / Double(X.count)
            }
            b -= lr * db / Double(X.count)
        }
    }
}

// =================== DEMO AND ===================
let X = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
let Y = [0.0,0.0,0.0,1.0]

let perceptron = Perceptron(nInputs: 2, lr: 0.1)
perceptron.train(X: X, Y: Y, epochs: 100)

print("=== Perceptron AND (Swift, activación lineal) ===\n")
for xi in X {
    let yhat = perceptron.predict(xi)        // salida lineal
    let label = yhat >= 0.5 ? 1 : 0          // binarización con threshold
    let desc = label == 1 ? "VERDADERO" : "FALSO"
    print("Entrada: \(xi) -> salida bruta: \(String(format: "%.4f", yhat)) -> clasificación: \(label) (\(desc))")
}
