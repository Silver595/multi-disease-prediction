// React App.js - Frontend for Disease Prediction

import React, { useState } from "react";
import axios from "axios";

const API_URL = "http://localhost:5000/api";

function App() {
  const [formData, setFormData] = useState({
    age: 45,
    gender: 1,
    bmi: 25.0,
    blood_pressure: 120,
    cholesterol: 200,
    blood_sugar: 100,
    heart_rate: 72,
    smoking: 0,
    exercise_hours: 3.0,
    family_history: 0,
  });

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: parseFloat(value) || parseInt(value),
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setResults(response.data.results);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Error making prediction");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Disease Risk Prediction System</h1>

      <form onSubmit={handleSubmit}>
        <div>
          <label>Age:</label>
          <input
            type="number"
            name="age"
            value={formData.age}
            onChange={handleChange}
          />
        </div>

        <div>
          <label>Gender (0=F, 1=M):</label>
          <input
            type="number"
            name="gender"
            value={formData.gender}
            onChange={handleChange}
          />
        </div>

        <div>
          <label>BMI:</label>
          <input
            type="number"
            step="0.1"
            name="bmi"
            value={formData.bmi}
            onChange={handleChange}
          />
        </div>

        {/* Add more form fields for other parameters */}

        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict Risk"}
        </button>
      </form>

      {results && (
        <div className="results">
          <h2>Prediction Results</h2>

          {Object.entries(results).map(([disease, data]) => (
            <div
              key={disease}
              className={`disease-card ${data.risk.toLowerCase()}`}
            >
              <h3>{disease.replace("_", " ").toUpperCase()}</h3>
              <p>Risk: {data.risk}</p>
              <p>Score: {data.score.toFixed(1)}%</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
