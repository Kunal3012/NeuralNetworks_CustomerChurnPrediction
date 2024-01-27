# Customer Churn Prediction using Neural Networks

## Introduction
In this project, we aim to predict customer churn in a credit card company using a neural network. The dataset used for this project is the "Churn_Modelling.csv" file, which contains information about customers, including features like credit score, geography, gender, age, tenure, balance, and more.

## Data Exploration and Preprocessing

### Data Overview
The dataset comprises 10,000 rows and 12 columns, including the target variable "Exited," indicating whether a customer has churned.

### Data Cleaning
The initial steps involved removing irrelevant columns ('RowNumber', 'CustomerId', 'Surname') and checking for missing values. Fortunately, the dataset was clean, with no null values.

### Descriptive Statistics
Statistical analysis revealed insightful information about the dataset, including mean, standard deviation, and quartiles for each numerical feature. This information provides a better understanding of the distribution and range of values.

## Feature Engineering

### One-Hot Encoding
Categorical variables ('Geography' and 'Gender') were one-hot encoded to enable compatibility with the neural network model.

## Data Splitting

### Train-Test Split
The dataset was split into training and testing sets with an 80-20 ratio using the `train_test_split` function from sklearn.

## Data Scaling

### Standardization
To ensure the neural network's optimal performance, numerical features were standardized using the StandardScaler from sklearn.

## Neural Network Model

### Model Architecture
The neural network model consists of an input layer, a hidden layer with 64 units and ReLU activation, another hidden layer with 32 units and sigmoid activation, and an output layer with one unit and sigmoid activation.

### Model Compilation
The model was compiled using the Adam optimizer and binary crossentropy loss. Additionally, accuracy and mean squared error were chosen as evaluation metrics.

### Model Training
The neural network was trained for 100 epochs with a batch size of 32, using both training and validation sets. The training history was recorded for further analysis.

## Model Evaluation

### Training Results
The training process resulted in progressively improving accuracy and decreasing loss. After 100 epochs, the model achieved satisfactory performance on both the training and validation sets.
![Training Outcomes](https://github.com/Kunal3012/NeuralNetworks_CustomerChurnPrediction/blob/main/training_outcome.png)

### Metrics
The final model's accuracy, mean squared error, and other metrics can be used to assess its performance.

## Conclusion

In this project, we successfully built and trained a neural network model to predict customer churn in a credit card company. The model demonstrated promising results during training, and further evaluation on test data will provide a comprehensive assessment of its predictive capabilities.

## Kaggle Code
Find the implementation code on Kaggle: [Neural Networks - Customer Churn Prediction](https://www.kaggle.com/kunal30122002/neuralnetworks-customerchurnprediction)

## Recommendations

- Further fine-tuning of hyperparameters may enhance the model's performance.
- Regular monitoring and updating of the model as new data becomes available are crucial for maintaining accuracy.

## Future Work

- Explore additional features that could potentially improve the model's predictive power.
- Experiment with different neural network architectures and algorithms for comparison.

The journey from data exploration to model training has provided valuable insights into customer churn prediction. The implemented neural network serves as a powerful tool for anticipating and mitigating potential customer losses.