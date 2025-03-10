# Importing the necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.preprocessing import MinMaxScaler  # To normalize data
from sklearn.feature_selection import RFE  # Recursive Feature Elimination (RFE) for feature selection
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.tree import DecisionTreeClassifier  # Decision Tree model
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.svm import SVC  # Support Vector Machine (SVM) model
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors (KNN) model
from sklearn.metrics import accuracy_score  # To calculate prediction accuracy

# Imports for building the neural network model with Keras
from tensorflow.keras.models import Sequential  # To create sequential models
from tensorflow.keras.layers import Dense, Dropout  # Fully connected (Dense) layers and Dropout for regularization
from tensorflow.keras.optimizers import Adam  # Adam optimizer for network training

# --------------------------------------------------------------------
# 1. Dataset Loading
# --------------------------------------------------------------------
# Loads the churn dataset from a CSV file. 
# This dataset contains customer information and whether they churned or not.
df = pd.read_csv(r'C:\Users\julia\Repos\juliano\TensorFlow\Churn.csv')

# --------------------------------------------------------------------
# 2. Data Preprocessing
# --------------------------------------------------------------------
# Processing the 'Total Charges' column:
# - Replaces empty values ('') with 0
df['Total Charges'] = df['Total Charges'].replace('', 0)

# - Fills null values (NaN) with 0
df['Total Charges'] = df['Total Charges'].fillna(0)

# - Converts the 'Total Charges' column to numeric (float), handling errors and replacing possible NaN with 0
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)

# --------------------------------------------------------------------
# 3. Normalization of Numerical Data
# --------------------------------------------------------------------
# Creates a MinMaxScaler object to normalize data between 0 and 1
scaler = MinMaxScaler()

# Normalizes the 'tenure', 'Monthly Charges', and 'Total Charges' columns
df[['tenure', 'Monthly Charges', 'Total Charges']] = scaler.fit_transform(
    df[['tenure', 'Monthly Charges', 'Total Charges']]
)

# --------------------------------------------------------------------
# 4. Data Preparation for Modeling
# --------------------------------------------------------------------
# Separates features (explanatory variables) and the target variable
# - Removes the 'Churn' column (which will be the target variable) and 'Customer ID' (not useful for prediction)
# - Converts categorical variables into dummy variables (one-hot encoding)
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))

# Creates the target variable 'y', transforming 'Yes' into 1 and 'No' into 0
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# --------------------------------------------------------------------
# 5. Splitting Data into Training and Testing Sets
# --------------------------------------------------------------------
# Splits data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------------------
# 6. Feature Selection Using RFE (Recursive Feature Elimination)
# --------------------------------------------------------------------
# Uses logistic regression as the base estimator for RFE
model_lr = LogisticRegression()

# Configures RFE to select the top 10 features based on their importance determined by the model
selector = RFE(model_lr, n_features_to_select=10)

# Fits (trains) RFE with the training data
selector = selector.fit(X_train, y_train)

# Identifies the column names selected by RFE
selected_columns = X_train.columns[selector.support_]
print("Selected columns:", selected_columns)

# Filters datasets to keep only the selected features
X_train_selected = X_train[selected_columns]
X_test_selected = X_test[selected_columns]

# --------------------------------------------------------------------
# 7. Function for Training and Evaluating Classic Models
# --------------------------------------------------------------------
def train_and_evaluate(model, model_name):
    """
    Trains a machine learning model, makes predictions on the test set, 
    and prints the model's accuracy.
    
    Parameters:
        model : the model object to be trained (e.g., LogisticRegression, DecisionTreeClassifier, etc.)
        model_name : name of the model (string) for displaying results
    """
    # Trains the model using the training data with selected features
    model.fit(X_train_selected, y_train)
    
    # Makes predictions on the test set
    y_hat = model.predict(X_test_selected)
    
    # Calculates accuracy by comparing predictions with actual test values
    accuracy = accuracy_score(y_test, y_hat)
    
    # Prints accuracy formatted to 4 decimal places
    print(f"Accuracy of the {model_name} model: {accuracy:.4f}")
    
    return accuracy

# --------------------------------------------------------------------
# 8. Training and Evaluating Classic Machine Learning Models
# --------------------------------------------------------------------
# Creates a dictionary containing the models to be tested
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# Iterates over each model, trains and evaluates, displaying the accuracy of each
for name, model in models.items():
    train_and_evaluate(model, name)

# --------------------------------------------------------------------
# 9. Building, Training, and Evaluating a Neural Network Model with Keras
# --------------------------------------------------------------------
# Creates a sequential model (linear stacking of layers)
model_nn = Sequential()

# Adds the first hidden layer:
# - 64 neurons
# - Activation function: 'relu'
# - 'input_dim' set to the number of selected features
model_nn.add(Dense(units=64, activation='relu', input_dim=len(X_train_selected.columns)))

# Adds a Dropout layer to reduce overfitting, randomly discarding 50% of neurons during training
model_nn.add(Dropout(0.5))

# Adds a second hidden layer with 128 neurons and 'relu' activation function
model_nn.add(Dense(units=128, activation='relu'))

# Adds the output layer:
# - 1 neuron, as this is a binary classification problem
# - Activation function: 'sigmoid' to produce an output between 0 and 1 (probability)
model_nn.add(Dense(units=1, activation='sigmoid'))

# Defines the Adam optimizer with a learning rate of 0.001
optimizer = Adam(learning_rate=0.001)

# Compiles the model specifying:
# - Loss function: 'binary_crossentropy', appropriate for binary classification
# - Optimizer: Adam
# - Metric: accuracy
model_nn.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Trains the neural network:
# - epochs: number of iterations over the dataset
# - batch_size: number of samples per weight update
# - validation_data: validation set to monitor performance during training
model_nn.fit(X_train_selected, y_train, epochs=200, batch_size=32, validation_data=(X_test_selected, y_test), verbose=1)

# Makes predictions on the test set using the neural network model
y_hat_nn = model_nn.predict(X_test_selected)

# Converts predictions (probabilities) into classes:
# If probability is less than 0.5, assign 0; otherwise, assign 1
y_hat_nn = [0 if val < 0.5 else 1 for val in y_hat_nn]

# Calculates and prints the neural network model's accuracy
print(f"Neural Network model accuracy: {accuracy_score(y_test, y_hat_nn):.4f}")
