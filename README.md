# Fraud Detection in Motor Insurance Sector

This project aims to build an RNN (Recurrent Neural Network) model for detecting fraud in the motor insurance sector. The model is implemented using Python and popular libraries such as TensorFlow, Keras, NumPy, Pandas, and Scikit-learn.

## Project Overview

In the insurance industry, fraudulent claims can lead to significant financial losses for companies. By leveraging machine learning techniques, we can develop models to identify potential fraudulent claims more accurately and efficiently. This project focuses on building an RNN model to detect fraud in motor insurance claims based on various features such as vehicle make, insured amount, claim history, and more.

## Dataset

The dataset used for this project contains information about motor insurance claims, including features such as kilometers driven, zone, bonus level, vehicle make, insured amount, number of claims, and total payment. The dataset is preprocessed and split into training and testing sets for model development and evaluation.

## Requirements

To run this project, you'll need the following:

- Python (version 3.6 or higher)
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn

You can install the required libraries using pip:

```
pip install tensorflow keras numpy pandas scikit-learn
```

## Implementation

The project follows these main steps:

1. **Data Preprocessing**: The dataset is loaded into a pandas DataFrame, and necessary preprocessing steps are performed, such as handling missing values, encoding categorical variables, and scaling the features.

2. **Data Splitting**: The preprocessed data is split into training and testing sets using the `train_test_split` function from Scikit-learn.

3. **Model Building**: An RNN model is built using the Sequential API from Keras. The model architecture consists of an SimpleRNN layer, followed by Dropout layers for regularization, and Dense layers for feature extraction and output prediction.

4. **Model Training**: The model is compiled with the Adam optimizer and binary cross-entropy loss function. It is then trained on the training data using the `fit` method, with validation data provided for monitoring performance during training.

5. **Model Evaluation**: After training, the model is evaluated on the test set using the `evaluate` method, which calculates the loss and accuracy metrics.

6. **Model Prediction**: Finally, the trained model can be used to make predictions on new data for fraud detection in motor insurance claims.

## Usage

1. Clone the repository or download the project files.
2. Ensure that you have the required libraries installed.
3. Run the Python script containing the implementation code.
4. The script will preprocess the data, train the RNN model, and evaluate its performance on the test set.
5. You can modify the code to experiment with different model architectures, hyperparameters, or preprocessing techniques.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
