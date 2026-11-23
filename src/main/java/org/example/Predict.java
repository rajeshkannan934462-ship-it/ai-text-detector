package org.example;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Predict {
    static final String BASE = "D:/dataset/ai-text-dedector/";
    private static Classifier model;
    private static Instances header;

    static {
        try {
            model = (Classifier) SerializationHelper.read(BASE + "models/detector.model");
            header = (Instances) SerializationHelper.read(BASE + "models/header.model");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static double[] predict(String text) throws Exception {
        Instances h = new Instances(header, 0);
        h.setClassIndex(1);
        DenseInstance inst = new DenseInstance(2);
        inst.setDataset(h);
        inst.setValue(h.attribute(0), text.replace("\"", "").replaceAll("\\s+", " ").trim());
        inst.setMissing(1);
        return model.distributionForInstance(inst);
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== Check 1: rows from data.csv ===");
        CSVFormat fmt = CSVFormat.DEFAULT.builder()
                .setHeader().setSkipHeaderRecord(true).build();
        int humanShown = 0;
        int aiShown = 0;

        try (Reader in = Files.newBufferedReader(Paths.get(BASE + "data/data.csv"));
             CSVParser parser = fmt.parse(in)) {
            for (CSVRecord r : parser) {
                String label = r.get("label");
                if (label.equals("human") && humanShown >= 5) continue;
                if (label.equals("ai") && aiShown >= 5) continue;
                if (label.equals("human")) humanShown++;
                else aiShown++;

                String text = r.get("text");
                double[] p = predict(text);
                int words = text.split(" ").length;
                System.out.printf("Actual: %-5s | words: %4d | Human: %5.1f%% | AI: %5.1f%%%n",
                        label, words, p[0] * 100, p[1] * 100);

                if (humanShown >= 5 && aiShown >= 5) break;
            }
        }

        System.out.println();
        System.out.println("=== Check 2: your own text from test.txt ===");
        Path file = Paths.get(BASE + "test.txt");
        if (Files.exists(file)) {
            String text = Files.readString(file);
            double[] p = predict(text);
            int words = text.trim().split("\\s+").length;
            System.out.printf("words: %d | Human: %.1f%% | AI: %.1f%%%n", words, p[0] * 100, p[1] * 100);
        } else {
            System.out.println("test.txt not found in " + BASE);
        }
    }
}