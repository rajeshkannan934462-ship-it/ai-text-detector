package org.example;

import org.apache.commons.csv.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Preparedata {
    static final String INPUT = "data/raw.csv";
    static final String TEXT_COL = "text";
    static final String LABEL_COL = "label";
    static final String AI_VALUE = "1";
    static final int MIN_WORDS = 30;
    static final int MAX_PER_CLASS = 3000;

    public static void main(String[] args) throws Exception {
        List<String> human = new ArrayList<>();
        List<String> ai = new ArrayList<>();
        Set<String> seen = new HashSet<>();

        CSVFormat fmt = CSVFormat.DEFAULT.builder()
                .setHeader().setSkipHeaderRecord(true).build();

        try (Reader in = Files.newBufferedReader(Paths.get(INPUT));
             CSVParser parser = fmt.parse(in)) {
            for (CSVRecord r : parser) {
                String text = r.get(TEXT_COL).replaceAll("\\s+", " ").trim();
                String label = r.get(LABEL_COL).trim();

                if (text.isEmpty()) continue;
                if (text.split(" ").length < MIN_WORDS) continue;
                if (!seen.add(text)) continue;

                if (label.equals(AI_VALUE)) ai.add(text);
                else human.add(text);
            }
        }

        System.out.println("Human: " + human.size() + " | AI: " + ai.size());

        // Balance classes
        Random rnd = new Random(42);
        Collections.shuffle(human, rnd);
        Collections.shuffle(ai, rnd);
        int n = Math.min(Math.min(human.size(), ai.size()), MAX_PER_CLASS);

        List<String[]> rows = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            rows.add(new String[]{human.get(i), "human"});
            rows.add(new String[]{ai.get(i), "ai"});
        }
        Collections.shuffle(rows, rnd);   // mix classes so the split is fair

        try (Writer out = Files.newBufferedWriter(Paths.get("data/data.csv"));
             CSVPrinter printer = new CSVPrinter(out,
                     CSVFormat.DEFAULT.builder().setHeader("text", "label").build())) {
            for (String[] row : rows) printer.printRecord(row[0], row[1]);
        }
        System.out.println("Saved data/data.csv with " + rows.size() + " rows");
    }
}