# 🏋️ Fitness Tracker – Machine Learning Project

An intelligent **Fitness Tracker** built with **Machine Learning** to automatically recognize strength training exercises, count repetitions, and analyze workout performance using wearable sensor data.

This project explores how **context-aware applications** can function as digital fitness assistants by using accelerometer and gyroscope data collected during real gym workouts.

---

## 📌 Project Motivation

Most commercial fitness trackers support walking, running, or cycling, but very few can accurately monitor **free weight strength training exercises**.

This project focuses on solving that problem by building a system that can:

- Detect gym exercises automatically  
- Count repetitions  
- Track performance  
- Analyze movement patterns  
- Detect improper exercise form  

Based on research in context-aware fitness systems. :contentReference[oaicite:0]{index=0}

---

## 🚀 Features

✅ Automatic Exercise Recognition  
✅ Repetition Counting  
✅ Exercise Form Detection  
✅ Machine Learning Classification  
✅ Sensor Data Processing  
✅ Strength Training Analytics  

---

## 🏋️ Supported Exercises

The model was trained on major barbell strength exercises:

- Bench Press  
- Deadlift  
- Overhead Press  
- Barbell Row  
- Squat  
- Rest / Idle State  

:contentReference[oaicite:1]{index=1}

---

## 🤖 Machine Learning Approach

This project uses supervised learning models trained on wearable motion sensor data.

### Models Evaluated:

- Random Forest ✅ Best Performance  
- Decision Tree  
- Support Vector Machine  
- K-Nearest Neighbors  
- Neural Network  
- Naive Bayes  

### Best Result:

**Random Forest Accuracy: 98.51%**

:contentReference[oaicite:2]{index=2}

---

## 📊 Sensor Data Used

Collected using wrist wearable sensors:

- Accelerometer (X, Y, Z)  
- Gyroscope (X, Y, Z)

Used to capture movement, orientation, and lifting patterns during exercises.

:contentReference[oaicite:3]{index=3}

---

## 🧠 Feature Engineering

Several advanced features were extracted:

- Low-pass filtering  
- Principal Component Analysis (PCA)  
- Time-domain statistics  
- Frequency-domain features (FFT)  
- Magnitude calculations  
- Clustering (K-Means)

These improved model performance significantly.

:contentReference[oaicite:4]{index=4}

---

## 🔢 Repetition Counting

A peak detection algorithm was used on filtered acceleration data.

### Result:

**Only ~5% counting error rate**

:contentReference[oaicite:5]{index=5}

---

## 🏆 Improper Form Detection

The system can also detect incorrect bench press execution such as:

- Bar too high on chest  
- Not touching chest properly  

### Accuracy:

**98.53%**

:contentReference[oaicite:6]{index=6}

---

## 🛠️ Technologies Used

### Languages
- Python

### Libraries
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- SciPy  

### Machine Learning
- Random Forest  
- Classification Models  
- Signal Processing  

---

## 📂 Project Structure

```bash
fitness-tracker/
│── data/               # Sensor datasets
│── notebooks/          # Experiments / Analysis
│── models/             # Trained models
│── src/                # Source code
│── visuals/            # Charts / Results
│── README.md
