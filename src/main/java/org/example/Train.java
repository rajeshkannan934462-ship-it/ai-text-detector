package org.example;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Train {
    static final String BASE = "D:/dataset/ai-text-dedector/";

    public static void main(String[] args) throws Exception {
        ArrayList<Attribute> attrs = new ArrayList<>();
        attrs.add(new Attribute("text", (List<String>) null));
        attrs.add(new Attribute("label", Arrays.asList("human", "ai")));

        Instances data = new Instances("essays", attrs, 0);
        data.setClassIndex(1);

        CSVFormat fmt = CSVFormat.DEFAULT.builder()
                .setHeader().setSkipHeaderRecord(true).build();

        try (Reader in = Files.newBufferedReader(Paths.get(BASE + "data/data.csv"));
             CSVParser parser = fmt.parse(in)) {
            for (CSVRecord r : parser) {
                DenseInstance inst = new DenseInstance(2);
                inst.setDataset(data);
                inst.setValue(data.attribute(0), r.get("text"));
                inst.setValue(data.attribute(1), r.get("label"));
                data.add(inst);
            }
        }

        System.out.println("Rows loaded: " + data.numInstances());

        data.randomize(new Random(42));
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);

        StringToWordVector tfidf = new StringToWordVector();
        tfidf.setLowerCaseTokens(true);
        tfidf.setTFTransform(true);
        tfidf.setIDFTransform(true);
        tfidf.setOutputWordCounts(true);
        tfidf.setWordsToKeep(5000);

        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(tfidf);
        fc.setClassifier(new NaiveBayesMultinomial());
        fc.buildClassifier(train);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(fc, test);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());

        Files.createDirectories(Paths.get(BASE + "models"));
        SerializationHelper.write(BASE + "models/detector.model", fc);
        SerializationHelper.write(BASE + "models/header.model", new Instances(train, 0));
        System.out.println("Model saved in: " + BASE + "models/");
    }
}