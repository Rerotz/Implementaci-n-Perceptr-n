package riesgo;

import java.io.*;
import java.util.*;

public class RiskCSVReader {
    private static List<double[]> features = new ArrayList<>();
    private static List<Integer> labels = new ArrayList<>();

    public static void load(String filename) throws IOException {
        features.clear();
        labels.clear();

        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;
        br.readLine(); // cabecera

        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");

            try {
                // 🔹 Tomamos 3 columnas numéricas
                double age = Double.parseDouble(parts[17]);   // edad
                double grade1 = Double.parseDouble(parts[22]); // nota 1
                double grade2 = Double.parseDouble(parts[28]); // nota 2

                // Target (33): "Dropout" = -1, "Graduate" = 1
                String target = parts[33].trim();
                int y = target.equalsIgnoreCase("Dropout") ? -1 : 1;

                features.add(new double[]{age, grade1, grade2});
                labels.add(y);
            } catch (Exception e) {
                // ignoramos filas inválidas
            }
        }
        br.close();

        normalizeZScore();
    }

    // 🔹 Normalización tipo z-score: (x - mean) / std
    private static void normalizeZScore() {
        if (features.isEmpty()) return;
        int nFeatures = features.get(0).length;
        double[] mean = new double[nFeatures];
        double[] std = new double[nFeatures];

        // calcular media
        for (double[] f : features) {
            for (int j = 0; j < nFeatures; j++) {
                mean[j] += f[j];
            }
        }
        for (int j = 0; j < nFeatures; j++) mean[j] /= features.size();

        // calcular desviación estándar
        for (double[] f : features) {
            for (int j = 0; j < nFeatures; j++) {
                std[j] += Math.pow(f[j] - mean[j], 2);
            }
        }
        for (int j = 0; j < nFeatures; j++) std[j] = Math.sqrt(std[j] / features.size());

        // aplicar normalización
        for (int i = 0; i < features.size(); i++) {
            double[] f = features.get(i);
            for (int j = 0; j < nFeatures; j++) {
                if (std[j] != 0) {
                    f[j] = (f[j] - mean[j]) / std[j];
                }
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

