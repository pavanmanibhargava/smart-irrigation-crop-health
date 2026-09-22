# Project Requirements

## 1. Project Overview

The **AI-Powered Smart Irrigation and Crop Health Prediction System** is a capstone software project that uses machine learning to:

- **Predict irrigation needs** based on environmental and soil conditions, and recommend optimal irrigation schedules to reduce water waste and improve crop yield.
- **Predict crop health** from leaf/plant images, classifying whether a crop is healthy or affected by disease.

The system provides a web-based **farm monitoring dashboard** where users can view current conditions, receive irrigation recommendations, inspect crop health predictions, and explore historical agricultural analytics.

During development, real sensor hardware is not available. The system will use **publicly available datasets** for ML model training and **software-simulated sensor data** for runtime demonstration. The architecture is designed so that real IoT sensors and edge devices (e.g., Raspberry Pi) can replace the simulator in a future phase without restructuring the codebase.

---

## 2. Smart Irrigation Module

### 2.1 Inputs

The irrigation model will accept environmental and soil-related inputs that influence irrigation decisions. Candidate input categories include:

- Soil moisture level
- Air temperature
- Relative humidity
- Rainfall or precipitation data
- Crop type or growth stage
- Additional environmental features

> **Note:** The exact feature set and column names are **to be determined during dataset selection**. The inputs listed above represent the conceptual categories the model is expected to use.

### 2.2 Prediction Output

The model will produce a prediction that indicates:

- **Whether irrigation is needed** (binary classification: irrigate / do not irrigate), OR
- **How much irrigation is needed** (regression: volume or duration)

> The precise formulation (classification vs. regression) is **to be determined during dataset selection** based on the target variable available.

### 2.3 Irrigation Recommendation

Based on the model's prediction and the current input conditions, the system will generate a human-readable recommendation such as:

- "Irrigation is recommended — soil moisture is below optimal range."
- "No irrigation needed — recent rainfall is sufficient."
- Suggested irrigation amount or duration (if the dataset supports a regression target).

The recommendation engine will combine the ML prediction with simple rule-based logic (e.g., thresholds on soil moisture) to produce actionable advice.

### 2.4 Data to Store

The system should persist:

- Each set of input sensor/environmental readings used for a prediction
- The model's prediction result
- The generated recommendation
- A timestamp for every record
- Metadata such as crop type (if applicable)

### 2.5 User-Facing Output

On the dashboard, the user should see:

- The latest environmental/soil readings
- The irrigation prediction (needed / not needed)
- The recommendation text with reasoning
- Historical irrigation predictions and recommendations

---

## 3. Crop Health Prediction Module

### 3.1 Input

The crop health model will accept **an image of a plant leaf or crop** as input. The image may be uploaded by the user through the dashboard or supplied programmatically.

### 3.2 Image Preprocessing

Before inference, each image must be preprocessed:

- Resize to the input dimensions expected by the model
- Normalize pixel values to a standard range (e.g., 0–1 or ImageNet normalization)
- Convert colour space if required by the model architecture
- Apply any augmentation used during training (center crop, etc.)

> The exact preprocessing pipeline is **to be determined** once a dataset and model architecture are selected.

### 3.3 Prediction Output

The model will output:

- A **predicted class label** (e.g., "Healthy", "Bacterial Blight", "Leaf Rust")
- A **confidence score** (probability associated with the predicted class)

### 3.4 Health / Disease Classification Concept

The module is conceptually a **multi-class image classification** task:

- One class represents a healthy crop
- Additional classes represent specific diseases or stress conditions
- The number and names of classes are **to be determined during dataset selection**

The model should be able to distinguish between healthy and unhealthy plants, and when unhealthy, identify the likely disease or condition.

### 3.5 Data to Store

The system should persist:

- The original uploaded image (or a reference/path to it)
- The predicted class label
- The confidence score
- A timestamp
- Any user-supplied metadata (crop type, field identifier, notes)

### 3.6 User-Facing Output

On the dashboard, the user should see:

- An image upload interface
- The prediction result displayed clearly (class label + confidence)
- A brief description or guidance for the predicted condition
- History of past predictions with thumbnails and results

---

## 4. Sensor / Data Management

The system will handle the following categories of agricultural data:

| Data Type | Description |
|---|---|
| **Soil moisture** | Water content in the soil, expressed as a percentage or volumetric measure |
| **Temperature** | Ambient air temperature (°C or °F) |
| **Humidity** | Relative humidity of the air (%) |
| **Rainfall / weather** | Precipitation amount, weather condition codes, or forecasts |
| **Crop information** | Crop type, growth stage, planting date |
| **Other environmental** | Wind speed, solar radiation, soil pH, soil type — as available |

### 4.1 Data Source Categories

The system distinguishes three levels of data sourcing:

#### Level 1 — Dataset-Generated Data (Training & Evaluation)

- Publicly available agricultural or irrigation datasets downloaded for ML model training and evaluation.
- These datasets provide the ground-truth feature values and labels.
- Used offline during the model development phase.

#### Level 2 — Simulated Sensor Data (Runtime Demonstration)

- Software-generated readings that mimic real sensor behaviour.
- Produced by a **sensor simulator module** that outputs realistic, time-varying values with configurable noise and seasonal patterns.
- Used at runtime to demonstrate the end-to-end system without physical hardware.

#### Level 3 — Future Real Sensor Data (IoT Integration)

- Readings from physical sensors (e.g., capacitive soil moisture sensors, DHT22 temperature/humidity sensors) connected to a Raspberry Pi or similar edge device.
- Data transmitted to the backend via HTTP or MQTT.
- Intended for a future phase; the system architecture should accommodate this source without major refactoring.

---

## 5. Farm Monitoring Dashboard

### 5.1 Overview

The dashboard is a web-based interface that gives the user a consolidated view of their farm's status. The first version (MVP) will be a **single-page or minimal multi-page application** focused on clarity and usability over visual complexity.

### 5.2 Dashboard Components

#### Current Sensor Readings

- Display the latest values for soil moisture, temperature, humidity, and rainfall.
- Values should be clearly labelled with units.
- Visual indicators (colour coding or icons) to show whether readings are within normal ranges.

#### Irrigation Status

- Show whether the system currently recommends irrigation (Yes / No).
- Display the timestamp of the most recent irrigation prediction.

#### Irrigation Recommendation

- Present the human-readable recommendation text.
- Include the key input values that influenced the recommendation.

#### Crop Health Prediction

- Provide an image upload area.
- Display the most recent prediction result (class label, confidence, uploaded image thumbnail).

#### Historical Charts

- Line or bar charts showing sensor reading trends over time.
- Irrigation prediction history (timeline of irrigate / do-not-irrigate decisions).
- Crop health prediction history (table or list view).

#### System Status

- Indicate whether the backend API is reachable.
- Show the data source mode (simulated vs. real sensors).
- Display the last data refresh timestamp.

#### Important Alerts

- Surface alerts when sensor readings cross critical thresholds (e.g., soil moisture critically low).
- Notify the user of consecutive irrigation recommendations that may indicate a problem.
- Display any system errors (API unreachable, model loading failure).

---

## 6. Agricultural Analytics

### 6.1 Types of Analytics

The system will provide the following analytics, limited to data that the system actually collects:

| Analytic | Description |
|---|---|
| **Sensor trends** | Time-series visualisations of soil moisture, temperature, humidity, and rainfall over selectable time windows (daily, weekly, monthly) |
| **Irrigation history** | Log and summary of all irrigation predictions and recommendations, including counts and frequency |
| **Prediction history** | Log of all crop health predictions with images, labels, and confidence scores |
| **Crop health statistics** | Aggregate counts and percentages of healthy vs. diseased classifications over time |
| **Water usage estimates** | Estimated water savings based on irrigation recommendations vs. a fixed-schedule baseline (if data supports it) |

> Analytics that require data not yet available (e.g., actual water consumption from flow meters, yield data) will **not** be included until the corresponding data source is integrated.

---

## 7. Database Requirements

### 7.1 Data to Store

The database will eventually store the following categories of information:

- **Sensor readings** — timestamped records of soil moisture, temperature, humidity, rainfall, and other environmental values, along with a flag indicating the data source (simulated or real).
- **Irrigation predictions** — the model's output for each prediction request, including input features, prediction result, recommendation text, and timestamp.
- **Crop health predictions** — each prediction's image reference, predicted label, confidence score, timestamp, and any user-supplied metadata.
- **System/configuration data** — active crop type, sensor simulation settings, model versioning, and user preferences (if applicable).
- **Alert history** — logged alerts with severity, message, timestamp, and acknowledgement status.

### 7.2 Design Principles

- Use a **relational database** (e.g., PostgreSQL or SQLite for the MVP) to enforce referential integrity.
- Design tables to support efficient time-range queries for analytics and dashboard charts.
- Store image files on the filesystem or object storage; keep only the file path/reference in the database.

> **The SQL schema will be designed in a later step.** This section defines *what* must be stored, not *how*.

---

## 8. Backend Requirements

### 8.1 Responsibilities

The backend server will be responsible for the following:

#### Data Access

- Provide API endpoints to read and write sensor readings, irrigation predictions, and crop health predictions.
- Support time-range filtering and pagination for historical queries.

#### ML Prediction

- Load trained irrigation and crop health models at startup.
- Accept prediction requests, run inference, and return results.
- Handle model versioning so that updated models can be deployed without downtime.

#### Recommendation Engine

- Apply rule-based logic on top of ML predictions to generate irrigation recommendations.
- Allow configurable thresholds (e.g., minimum soil moisture before recommending irrigation).

#### Dashboard Data

- Serve aggregated and formatted data for the dashboard's components (latest readings, charts, statistics).
- Provide a system status/health endpoint.

#### Validation

- Validate all incoming request payloads (data types, value ranges, required fields).
- Validate uploaded images (file type, size limits, dimensions).

#### Error Handling

- Return consistent, structured error responses (status code, error message, details).
- Log errors with sufficient context for debugging.
- Gracefully handle model loading failures, database connection issues, and invalid inputs.

### 8.2 Technology Note

The backend will be built with **Python** and **FastAPI** (or equivalent async framework). Implementation will begin in a later step.

---

## 9. Simulation Requirements

### 9.1 Purpose

Because physical sensors and IoT hardware are not currently available, the system requires a **software sensor simulator** that generates realistic agricultural data for demonstration and testing.

### 9.2 Simulated Data Types

The simulator should generate time-series values for:

| Parameter | Typical Range | Behaviour |
|---|---|---|
| **Soil moisture** | 10–80% | Gradually decreases over time; resets higher after simulated irrigation or rain |
| **Temperature** | 5–45 °C | Follows a diurnal (day/night) cycle with seasonal variation |
| **Humidity** | 20–100% | Inversely correlated with temperature; higher at night and after rain |
| **Rainfall** | 0–50 mm/hr | Intermittent; modelled as random events with configurable probability and intensity |

### 9.3 Simulator Properties

- Generate readings at configurable intervals (e.g., every 5 minutes, every hour).
- Include controlled random noise to mimic sensor imprecision.
- Support deterministic mode (fixed random seed) for reproducibility during testing.
- Allow parameter overrides to simulate specific scenarios (drought, heavy rain, equipment failure).

### 9.4 Integration

- The simulator will expose the same data interface as a real sensor data source.
- Switching between simulated and real data should require only a configuration change, not a code change.

> **The simulator will be implemented in a later step.** This section defines the behavioural requirements only.

---

## 10. Future Hardware Integration

### 10.1 Vision

In a future phase, the software sensor simulator will be replaced (or supplemented) by real hardware:

- **Microcontroller / edge device:** Raspberry Pi (or similar) running a lightweight data-collection script.
- **Sensors:** Capacitive soil moisture sensor, DHT22 (temperature + humidity), rain gauge, and optional sensors (light intensity, soil pH).
- **Communication:** The edge device will send readings to the backend via HTTP REST calls or an MQTT broker.

### 10.2 Architectural Considerations

To support a smooth transition:

- The backend's data ingestion layer will accept sensor data through a well-defined API contract, regardless of whether the source is the simulator or a real device.
- Sensor readings will include a `source` field (`simulated` | `hardware`) so the system can distinguish origin.
- The Raspberry Pi script will be a thin client that reads sensor values and POSTs them to the backend — no ML inference on the edge device in the MVP.

### 10.3 Current Scope

- **No hardware will be purchased, configured, or programmed during the current phase.**
- The architecture is designed to accommodate hardware integration later with minimal refactoring.

---

## 11. Non-Functional Requirements

### 11.1 Maintainable Code

- Follow consistent coding style (PEP 8 for Python, ESLint/Prettier for JavaScript/TypeScript).
- Use meaningful variable, function, and module names.
- Include docstrings and inline comments where logic is non-obvious.

### 11.2 Modular Architecture

- Separate concerns into distinct modules: data access, ML models, recommendation logic, API layer, simulation, and frontend.
- Each module should be independently testable and replaceable.
- Avoid tight coupling between the ML pipeline and the API layer.

### 11.3 Reproducibility

- Pin all dependency versions in `requirements.txt` (Python) and `package.json` (frontend).
- Use fixed random seeds for model training and data simulation to ensure reproducible results.
- Document the environment setup and data preparation steps.

### 11.4 Version Control

- Use **Git** for version control with the repository hosted on **GitHub**.
- Follow a branching strategy (e.g., feature branches merged via pull requests).
- Write clear, descriptive commit messages.
- Use `.gitignore` to exclude virtual environments, data files, model binaries, and OS-generated files.

### 11.5 Testing

- Write **unit tests** for core logic (prediction functions, recommendation engine, data validation).
- Write **integration tests** for API endpoints.
- Aim for meaningful test coverage on critical paths; 100% coverage is not required for the MVP.
- Use `pytest` for Python tests.

### 11.6 Documentation

- Maintain an up-to-date `README.md` with setup instructions.
- Document API endpoints (manually or via auto-generated docs such as FastAPI's Swagger UI).
- Keep this requirements document and future architecture/design documents current as the project evolves.

---

## 12. MVP Scope

### 12.1 Definition

The **Minimum Viable Product (MVP)** is the smallest working version of the system that demonstrates the core value proposition end-to-end. The MVP must function entirely in software — **no physical hardware is required**.

### 12.2 MVP Features

| Component | MVP Deliverable |
|---|---|
| **Irrigation prediction** | A trained ML model that accepts environmental/soil inputs and predicts irrigation need |
| **Irrigation recommendation** | A recommendation engine that converts the prediction into actionable advice |
| **Crop health prediction** | A trained image classification model that predicts crop health/disease from a leaf image |
| **Database** | A relational database storing sensor readings, predictions, recommendations, and alerts |
| **Backend API** | A FastAPI server exposing endpoints for predictions, data retrieval, and dashboard data |
| **Dashboard** | A basic web frontend displaying current readings, predictions, recommendations, and simple charts |
| **Sensor simulation** | A simulator generating realistic soil moisture, temperature, humidity, and rainfall data |

### 12.3 Explicitly Out of MVP Scope

The following are **not** required for the MVP but are planned for future iterations:

- Real IoT hardware and Raspberry Pi integration
- User authentication and multi-tenant support
- Mobile application
- Real-time streaming (WebSockets) — polling is acceptable for the MVP
- Advanced analytics (yield prediction, economic analysis)
- Deployment to cloud infrastructure (local development is sufficient)
- CI/CD pipelines
- Multi-language / internationalisation support
