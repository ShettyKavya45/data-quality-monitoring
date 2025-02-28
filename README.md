# AI-Driven Data Quality Monitoring
# Automating Data Quality Monitoring in Cloud Database Warehouses with AI

## 📌 Project Overview
This project automates **data quality monitoring** in cloud databases using **AI techniques**. The system identifies and resolves issues such as **missing data, outliers, and inconsistencies** while providing **a real-time dashboard for analysis and monitoring**.

## 🛠️ Features
- **Data Cleaning** (Handles missing values, detects outliers, balances data)
- **Machine Learning Models** (Decision Tree, Random Forest, and AutoAI for evaluation)
- **Confusion Matrix, ROC Curve & Performance Metrics**
- **Flask Web Interface** (Upload CSV, analyze data, view results in a dashboard)
- **Real-Time Monitoring on IBM Cloud with AutoAI**
- **Automated Alerts & API Integration**

## 🚀 Setup & Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/ShettyKavya45/AI-Data-Quality-Monitoring.git
cd AI-Data-Quality-Monitoring
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App
```sh
python app.py
```

### 4️⃣ Open in Browser
Visit:
```
http://127.0.0.1:5000/
```
Upload a CSV file and view results!

## 📂 Project Structure
```
AI_Data_Quality_Project/
│── templates/
│   ├── upload.html        # Upload Page
│   ├── dashboard.html     # Dashboard Page
│── uploads/               # Stores uploaded files and images
│── static/                # Stores visualization images
│── app.py                 # Flask application
│── requirements.txt        # Dependencies
│── README.md              # Project details
```

## 📊 Results & Visualizations
### Model Performance
| Model          | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|----------|--------|----------|
| Decision Tree | 85.2%    | 82.5%    | 84.1%  | 83.2%    |
| Random Forest | 92.1%    | 90.3%    | 91.7%  | 90.9%    |
| AutoAI Model  | 95.6%    | 94.8%    | 95.1%  | 94.9%    |

### Visualizations
- **ROC Curve**: Displays the trade-off between true positive and false positive rates.
- **Confusion Matrix**: Highlights correctly and incorrectly classified data quality issues.

## 📊 Real-Time Dashboard Features
- **Live Monitoring**: Displays data quality metrics in real-time.
- **Flagged Data Issues**: Highlights missing values, anomalies, and inconsistencies.
- **Interactive Visualizations**: Graphs showing trends in data quality issues over time.
- **Automated Alerts**: Notifies users of critical data quality problems.

## ☁️ Deploying AutoAI Model on IBM Cloud
### Steps to Deploy
1. **Create an IBM Cloud Account** at [IBM Cloud](https://cloud.ibm.com/).
2. **Deploy AutoAI Model**:
   - Navigate to Watson Studio and create a new project.
   - Upload the trained model.
   - Deploy it as a REST API for integration with the dashboard.
3. **Generate API Key** for secure communication with the deployed model.



## 📈 Resource Utilization on IBM Cloud
| Resource         | Free Tier Usage | Optimized Usage |
|-----------------|----------------|-----------------|
| Compute Hours   | 10 CUH/month    | 50 CUH/month    |
| API Calls       | 50 calls/month  | 500 calls/month |
| Storage        | 1GB (Lite)      | 10GB (Paid)     |

## 🔮 Future Enhancements
- **Deploying the UI on Vercel** for scalable hosting.
- **Allowing CSV Uploads** so users can analyze custom datasets.
- **Database Integration** to store historical data quality reports.
- **Automated Report Generation** to export summary insights in PDF.

## 📝 Contributing
Feel free to **fork** this project and submit pull requests!

## 📜 License
This project is licensed under the **MIT License**.



