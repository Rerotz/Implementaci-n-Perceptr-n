package fraude;

import java.io.*;
import java.util.*;

public class FraudCSVReader {
    public static List<double[]> features = new ArrayList<>();
    public static List<Integer> labels = new ArrayList<>();

    public static void load(String filename) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;
        br.readLine(); // cabecera
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            // Usamos algunas columnas numéricas: amount, oldbalanceOrg, newbalanceOrig
            double amount = Double.parseDouble(parts[2]);
            double oldBal = Double.parseDouble(parts[4]);
            double newBal = Double.parseDouble(parts[5]);
            int isFraud = Integer.parseInt(parts[9]);
            features.add(new double[]{amount, oldBal, newBal});
            labels.add(isFraud);
        }
        br.close();

        // 🔹 Normalizar después de cargar
        normalize();
    }

    private static void normalize() {
        for (int i = 0; i < features.size(); i++) {
            double[] f = features.get(i);
            for (int j = 0; j < f.length; j++) {
                f[j] = f[j] / 100000.0;  // escala simple
            }
            features.set(i, f);
        }
    }

    public static double[][] getX() {
        return features.toArray(new double[0][]);
    }

    public static int[] getY() {
        return labels.stream().mapToInt(Integer::intValue).toArray();
    }
}
