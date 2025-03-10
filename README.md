# Machine Learning Portfolio: Churn Prediction  

This project demonstrates the application of Machine Learning and Deep Learning techniques to predict churn (customer cancellation). It explores various essential steps, from data preprocessing to evaluating different classification models.  

## 📌 Project Description  
The data was extracted from this Kaggle dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  

The goal of this project is to compare the performance of various classification algorithms for the churn problem. The script performs the following steps:  

### 🔹 Data Processing  
- **Data Loading and Preprocessing**: Reads the dataset, handles missing values, and normalizes the data.  
- **Data Preparation**: Converts categorical variables into dummy variables and separates the target variable.  
- **Feature Selection**: Uses RFE (Recursive Feature Elimination) to select the top 10 features.  

### 🔹 Model Training  
- **Classical Models**: Tests algorithms such as Logistic Regression, Decision Tree, Random Forest, SVM, and KNN.  
- **Neural Network Model**: Builds, trains, and evaluates a neural network using Keras.  

### 🔹 Results Evaluation  
- Displays the accuracy of each model to facilitate performance analysis.  

## ⚙️ Prerequisites  

To run this project, you will need to have the following installed:  

### 🛠️ Required Tools  
- **Python 3.9.1**  
- **Libraries**:  
  - `pandas`  
  - `scikit-learn`  
  - `tensorflow` (includes Keras)  

---

## 📝 Script Details  

The code is organized into the following sections:  

### 🔹 1. Dataset Loading  
- Reads the CSV file and loads the data into a Pandas DataFrame.  

### 🔹 2. Preprocessing  
- Handles missing values and converts the `'Total Charges'` column to a numeric type.  
- Normalizes numerical columns using `MinMaxScaler`.  

### 🔹 3. Data Preparation  
- Converts categorical variables into dummy variables.  
- Separates the target variable (**Churn**) and transforms it into binary values (0 = "No", 1 = "Yes").  

### 🔹 4. Data Splitting  
- Splits the dataset into training and testing sets (80%/20%).  

### 🔹 5. Feature Selection with RFE  
- Uses logistic regression to select the top 10 features from the dataset.  

### 🔹 6. Training Classical Models  
- Trains and evaluates models:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - SVM  
  - KNN  
- Displays the accuracy of each model.  

### 🔹 7. Neural Network Model with Keras  
- Builds a neural network with two hidden layers and Dropout regularization.  
- Trains and evaluates performance on the test set.  

---

## 📊 Results  

After running the script, the accuracy of different models will be displayed in the terminal. This comparison helps identify which approach best fits the churn problem for this dataset.  

---

## 🤝 Contributions  

Contributions are welcome! If you would like to improve this project, feel free to:  

- 🐞 Open **issues** to report bugs or suggest improvements.  
- ✨ Submit **pull requests** with new features or fixes.  

---

## 📜 License  

This project is licensed under the **MIT License**.  

---

## 📬 Contact  

If you have any questions or suggestions, feel free to reach out via **GitHub Issues** or email.  

🚀 Happy Coding!  
