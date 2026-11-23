package org.example;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class DetectController {

    public record TextRequest(String text) {}

    @PostMapping("/predict")
    public ResponseEntity<?> predict(@RequestBody TextRequest req) {
        String text = req.text() == null ? "" : req.text().trim();
        int words = text.isEmpty() ? 0 : text.split("\\s+").length;

        if (words < 30) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "please Enter Maximum 30 Words Now you have only entered: " + words));
        }
        if (text.length() > 20000) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "Text is too big. Give 20,000 characters Below."));
        }

        try {
            double[] p = Predict.predict(text);
            double human = Math.round(p[0] * 1000) / 10.0;
            double ai = Math.round(p[1] * 1000) / 10.0;
            return ResponseEntity.ok(Map.of("human", human, "ai", ai, "words", words));
        } catch (Throwable e) {
            return ResponseEntity.internalServerError()
                    .body(Map.of("error", "Prediction failed: " + e.getMessage()));
        }
    }
}