const express = require("express");
const bodyParser = require("body-parser");
const { exec } = require("child_process");
const path = require("path");

const app = express();

app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, "../frontend")));

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/index.html"));
});

app.post("/predict", (req, res) => {
  const inputData = JSON.stringify(req.body);

  exec(`python ../model_api/predict.py '${inputData}'`, (err, stdout, stderr) => {
    if (err) {
      console.error("❌ Python Error:", stderr);
      return res.status(500).json({ error: "Prediction failed" });
    }

    res.json({ prediction: stdout.trim() });
  });
});

app.listen(3000, () => {
  console.log("🔥 Server running on http://localhost:3000");
});