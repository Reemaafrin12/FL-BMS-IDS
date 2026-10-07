import "dotenv/config";
import cors from "cors";
import express from "express";
import mongoose from "mongoose";

const { MONGODB_URI, ML_SERVICE_URL = "http://127.0.0.1:8000", PORT = 5000 } = process.env;
if (!MONGODB_URI) throw new Error("MONGODB_URI is required; copy .env.example to .env.");

const readingSchema = new mongoose.Schema({
  timestamp: { type: Date, default: Date.now },
  shunt_voltage: { type: Number, required: true },
  bus_voltage_V: { type: Number, required: true },
  current_mA: { type: Number, required: true },
  power_mW: { type: Number, required: true },
  State: { type: String, required: true },
  prediction: { type: String, enum: ["attack", "benign"], required: true },
  confidence: { type: Number, required: true },
});
const Prediction = mongoose.model("Prediction", readingSchema);

const app = express();
app.use(cors());
app.use(express.json());

const featureNames = ["shunt_voltage", "bus_voltage_V", "current_mA", "power_mW", "State"];

app.post("/api/readings", async (req, res, next) => {
  try {
    const reading = Object.fromEntries(featureNames.map((key) => [key, req.body[key]]));
    if (featureNames.some((key) => reading[key] === undefined)) {
      return res.status(400).json({ error: `Expected: ${featureNames.join(", ")}` });
    }
    const predictionResponse = await fetch(`${ML_SERVICE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reading),
    });
    if (!predictionResponse.ok) {
      return res.status(502).json({ error: "Prediction service failed", detail: await predictionResponse.text() });
    }
    const { prediction, confidence } = await predictionResponse.json();
    const saved = await Prediction.create({ ...reading, prediction, confidence });
    return res.status(201).json(saved);
  } catch (error) {
    next(error);
  }
});

app.get("/api/readings/recent", async (_req, res, next) => {
  try {
    res.json(await Prediction.find().sort({ timestamp: -1 }).limit(50).lean());
  } catch (error) {
    next(error);
  }
});

app.use((error, _req, res, _next) => {
  console.error(error);
  res.status(500).json({ error: "Internal server error" });
});

await mongoose.connect(MONGODB_URI);
app.listen(PORT, () => console.log(`Dashboard API listening on http://127.0.0.1:${PORT}`));
