package riesgo;

public class MainRiesgo {
    public static void main(String[] args) throws Exception {
        RiskCSVReader.load("C:\\Users\\Usuario\\Documents\\NetBeansProjects\\RiesgoTanh\\src\\main\\java\\riesgo\\riesgo.csv");
        double[][] X = RiskCSVReader.getX();
        int[] y = RiskCSVReader.getY();

        PerceptronTanh model = new PerceptronTanh(3); // usamos 3 features
        model.train(X, y, 0.01, 500);

        // Balance de clases
        int count0 = 0, count1 = 0;
        for (int label : y) {
            if (label == 0) count0++;
            else count1++;
        }
        System.out.println("Clase 0 (Graduate): " + count0);
        System.out.println("Clase 1 (Dropout): " + count1);

        // Accuracy
        int correct = 0;
        for (int i = 0; i < X.length; i++) {
            if (model.predict(X[i]) == y[i]) correct++;
        }
        System.out.println("Accuracy riesgo académico: " + (100.0 * correct / X.length) + "%");

        // Ejemplos
        System.out.println("\n=== Ejemplos de salida ===");
        for (int i = 0; i < 5 && i < X.length; i++) {
            double[] entrada = X[i];
            int esperado = y[i];
            double salida = model.rawOutput(entrada);
            int pred = model.predict(entrada);

            System.out.printf("Ejemplo %d:\n", i+1);
            System.out.printf(" Entrada: [%.2f, %.2f, %.2f]\n", entrada[0], entrada[1], entrada[2]);
            System.out.println(" Salida esperada: " + esperado);
            System.out.printf(" Salida tanh bruta: %.5f\n", salida);
            System.out.println(" Predicción final: " + pred + "\n");
        }
    }
}
