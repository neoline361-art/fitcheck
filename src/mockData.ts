export const mockDatasets = {
  creditRisk: {
    name: "Noisy Credit Risk Dataset",
    filename: "credit_risk_raw.csv",
    target: "Defaulted",
    description: "Contains missing incomes, outlier debt ratios, duplicated applications, and target imbalance.",
    data: [
      { ClientID: 1001, Age: 34, AnnualIncome: 65000, DebtRatio: 0.12, Defaulted: 0 },
      { ClientID: 1002, Age: 45, AnnualIncome: null, DebtRatio: 0.25, Defaulted: 0 }, // missing income
      { ClientID: 1003, Age: 23, AnnualIncome: 34000, DebtRatio: 12.5, Defaulted: 1 }, // outlier debt
      { ClientID: 1004, Age: 52, AnnualIncome: 110000, DebtRatio: 0.05, Defaulted: 0 },
      { ClientID: 1005, Age: 29, AnnualIncome: 48000, DebtRatio: 0.35, Defaulted: 0 },
      { ClientID: 1005, Age: 29, AnnualIncome: 48000, DebtRatio: 0.35, Defaulted: 0 }, // Duplicate Row
      { ClientID: 1006, Age: 31, AnnualIncome: 58000, DebtRatio: 0.18, Defaulted: 0 },
      { ClientID: 1007, Age: 38, AnnualIncome: 72000, DebtRatio: 0.22, Defaulted: 0 },
      { ClientID: 1008, Age: 41, AnnualIncome: 85000, DebtRatio: 0.15, Defaulted: 0 },
      { ClientID: 1009, Age: 26, AnnualIncome: 42000, DebtRatio: 0.31, Defaulted: 0 },
      { ClientID: 1010, Age: 50, AnnualIncome: 125000, DebtRatio: 0.08, Defaulted: 0 },
      { ClientID: 1011, Age: 47, AnnualIncome: null, DebtRatio: 0.45, Defaulted: 0 }, // missing income
      { ClientID: 1012, Age: 35, AnnualIncome: 67000, DebtRatio: 0.19, Defaulted: 0 },
      { ClientID: 1013, Age: 28, AnnualIncome: 49000, DebtRatio: 0.27, Defaulted: 0 },
      { ClientID: 1014, Age: 60, AnnualIncome: 140000, DebtRatio: 0.04, Defaulted: 0 },
      { ClientID: 1015, Age: 32, AnnualIncome: 55000, DebtRatio: 18.2, Defaulted: 0 }, // outlier debt
      { ClientID: 1016, Age: 44, AnnualIncome: 89000, DebtRatio: 0.11, Defaulted: 0 },
      { ClientID: 1017, Age: 39, AnnualIncome: 75000, DebtRatio: 0.21, Defaulted: 0 },
      { ClientID: 1018, Age: 43, AnnualIncome: 92000, DebtRatio: 0.14, Defaulted: 0 },
      { ClientID: 1019, Age: 25, AnnualIncome: 39000, DebtRatio: 0.38, Defaulted: 0 },
    ],
  },
  engineTelemetry: {
    name: "Engine Telemetry Logs",
    filename: "engine_diagnostics.csv",
    target: "status",
    description: "Contains fixed-variance static sensors and anomalous thermal/vibration outliers.",
    data: [
      { Timestamp: "12:00:00", Temp_C: 75.2, Vibration_Hz: 45, Sensor_Static_M1: 99.9, status: "Normal" },
      { Timestamp: "12:01:00", Temp_C: 76.1, Vibration_Hz: 46, Sensor_Static_M1: 99.9, status: "Normal" },
      { Timestamp: "12:02:00", Temp_C: 75.8, Vibration_Hz: 44, Sensor_Static_M1: 99.9, status: "Normal" },
      { Timestamp: "12:03:00", Temp_C: 382.5, Vibration_Hz: 45, Sensor_Static_M1: 99.9, status: "Normal" }, // Outlier temp
      { Timestamp: "12:04:00", Temp_C: 75.4, Vibration_Hz: 45, Sensor_Static_M1: 99.9, status: "Normal" },
      { Timestamp: "12:05:00", Temp_C: 76.0, Vibration_Hz: 198, Sensor_Static_M1: 99.9, status: "Anomaly" }, // Outlier vibration
      { Timestamp: "12:06:00", Temp_C: 75.9, Vibration_Hz: 46, Sensor_Static_M1: 99.9, status: "Normal" },
      { Timestamp: "12:07:00", Temp_C: 76.3, Vibration_Hz: 45, Sensor_Static_M1: 99.9, status: "Normal" },
      { Timestamp: "12:08:00", Temp_C: 75.5, Vibration_Hz: 44, Sensor_Static_M1: 99.9, status: "Normal" },
      { Timestamp: "12:09:00", Temp_C: 76.1, Vibration_Hz: 45, Sensor_Static_M1: 99.9, status: "Normal" },
    ],
  },
  perfectIris: {
    name: "Iris Species Profile (Clean)",
    filename: "iris_dataset_clean.csv",
    target: "Species",
    description: "A standard benchmark dataset completely clean of missing data, constants, or outliers.",
    data: [
      { SepalLength: 5.1, SepalWidth: 3.5, PetalLength: 1.4, PetalWidth: 0.2, Species: "setosa" },
      { SepalLength: 4.9, SepalWidth: 3.0, PetalLength: 1.4, PetalWidth: 0.2, Species: "setosa" },
      { SepalLength: 4.7, SepalWidth: 3.2, PetalLength: 1.3, PetalWidth: 0.2, Species: "setosa" },
      { SepalLength: 4.6, SepalWidth: 3.1, PetalLength: 1.5, PetalWidth: 0.2, Species: "setosa" },
      { SepalLength: 5.0, SepalWidth: 3.6, PetalLength: 1.4, PetalWidth: 0.2, Species: "setosa" },
      { SepalLength: 7.0, SepalWidth: 3.2, PetalLength: 4.7, PetalWidth: 1.4, Species: "versicolor" },
      { SepalLength: 6.4, SepalWidth: 3.2, PetalLength: 4.5, PetalWidth: 1.5, Species: "versicolor" },
      { SepalLength: 6.9, SepalWidth: 3.1, PetalLength: 4.9, PetalWidth: 1.5, Species: "versicolor" },
      { SepalLength: 5.5, SepalWidth: 2.3, PetalLength: 4.0, PetalWidth: 1.3, Species: "versicolor" },
      { SepalLength: 6.5, SepalWidth: 2.8, PetalLength: 4.6, PetalWidth: 1.5, Species: "versicolor" },
      { SepalLength: 6.3, SepalWidth: 3.3, PetalLength: 6.0, PetalWidth: 2.5, Species: "virginica" },
      { SepalLength: 5.8, SepalWidth: 2.7, PetalLength: 5.1, PetalWidth: 1.9, Species: "virginica" },
      { SepalLength: 7.1, SepalWidth: 3.0, PetalLength: 5.9, PetalWidth: 2.1, Species: "virginica" },
      { SepalLength: 6.3, SepalWidth: 2.9, PetalLength: 5.6, PetalWidth: 1.8, Species: "virginica" },
      { SepalLength: 6.5, SepalWidth: 3.0, PetalLength: 5.8, PetalWidth: 2.2, Species: "virginica" },
    ],
  },
};

// Drift distributions samples
export const mockDriftData = {
  reference: [
    { age: 25, income: 45000, rating: "A" },
    { age: 28, income: 48000, rating: "B" },
    { age: 34, income: 52000, rating: "A" },
    { age: 22, income: 41000, rating: "C" },
    { age: 45, income: 72000, rating: "A" },
    { age: 31, income: 55000, rating: "B" },
    { age: 29, income: 50000, rating: "A" },
    { age: 36, income: 61000, rating: "B" },
    { age: 40, income: 68000, rating: "B" },
    { age: 38, income: 64000, rating: "A" },
    { age: 27, income: 46000, rating: "A" },
    { age: 33, income: 54000, rating: "B" },
    { age: 42, income: 70000, rating: "A" },
    { age: 30, income: 51000, rating: "C" },
    { age: 48, income: 82000, rating: "A" },
  ],
  production: [
    // Age remains stable, but Income has drifted upwards significantly, rating has shifted too.
    { age: 24, income: 85000, rating: "C" },
    { age: 29, income: 92000, rating: "C" },
    { age: 35, income: 110000, rating: "B" },
    { age: 23, income: 79000, rating: "C" },
    { age: 46, income: 145000, rating: "B" },
    { age: 30, income: 101000, rating: "C" },
    { age: 28, income: 88000, rating: "C" },
    { age: 37, income: 115000, rating: "C" },
    { age: 41, income: 130000, rating: "B" },
    { age: 39, income: 121000, rating: "C" },
    { age: 26, income: 89000, rating: "C" },
    { age: 32, income: 98000, rating: "B" },
    { age: 43, income: 135000, rating: "C" },
    { age: 31, income: 104000, rating: "C" },
    { age: 47, income: 155000, rating: "B" },
  ],
};

// Model evaluation datasets
export const mockEvaluationData = {
  classification: {
    actuals:   [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    predictions: [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]
  },
  regression: {
    actuals:   [250000, 310000, 185000, 420000, 290000, 510000, 360000, 210000, 450000, 280000],
    predictions: [242000, 315000, 192000, 405000, 278000, 498000, 375000, 205000, 438000, 289000]
  }
};
