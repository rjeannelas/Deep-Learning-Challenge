# Deep-Learning-Challenge
### Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

##### TOOLS: Google Colab

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
 - What variable(s) are the target(s) for your model?
 - What variable(s) are the feature(s) for your model?
2. Drop the EIN and NAME columns.
3. Determine the number of unique values for each column.
4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
6. Use pd.get_dummies() to encode categorical variables.
7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### Step 2: Compile, Train, and Evaluate the Model
Design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset.
 - How many inputs there are before determining the number of neurons and layers in your model?
Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

##### TOOLS: TensorFlow

1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3. Create the first hidden layer and choose an appropriate activation function.
** If necessary, add a second hidden layer with an appropriate activation function.
4. Create an output layer with an appropriate activation function.
5. Check the structure of the model.
6. Compile and train the model.
7. Create a callback that saves the model's weights every five epochs.
8. Evaluate the model using the test data to determine the loss and accuracy.
9. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

### Step 3: Optimize the Model
Optimize your model to achieve a target predictive accuracy higher than 75%.
Use any or all of the following methods to optimize your model:
- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
- Dropping more or fewer columns.
- Creating more bins for rare occurrences in columns.
- Increasing or decreasing the number of values for each bin.
- Add more neurons to a hidden layer.
- Add more hidden layers.
- Use different activation functions for the hidden layers.
- Add or reduce the number of epochs to the training regimen.

1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.
3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

### Step 4: Report on the Neural Network Model
###### Overview of the analysis:
- The purpose of this model to use a tool that can help select the applicants for funding with the best chance of success in their ventures. Using machine learning and the neural network model to create a binary classifier that can predict the probability of applicants from more than 34,000 organizations will be successful if funded by Alphabet Soup.

###### Results:
Using bulleted lists and images to support your answers, address the following questions:
Data Preprocessing
- What variable(s) are the target(s) for your model?
 - The target variable is the IS_SUCCESSFUL column in the dataset, with a 1 or a 0 meaning successful or unsuccessful use of funding 
- What variable(s) are the features for your model?
 - The IS_SUCCESSFUL, EIN, and NAME columns were dropped so that the remaining columns serve as variables/features in the subsequent models
- What variable(s) should be removed from the input data because they are neither targets nor features?
 - Specific variables were not removed from the dataset. By using the Lasso and Ridge tests to reduce the data size, I was able to retain much of the explained variances and set the boundries.

Compiling, Training, and Evaluating the Model
- How many neurons, layers, and activation functions did you select for your neural network model, and why?
 - Continued to use default model parameters of 80 neurons on the initial layer, 30 for the hidden layer, and 1 for the output layer with relu, relu, sigmoid
 - Using Keras Tuner trials, limiting activation functions that did not improve the model by eliminating them from the pool. This helped find the optimal selection of neurons, layers, and activation functions.
- Were you able to achieve the target model performance?
 -  The target model performance manage to have a predictive accuracy of 75.5% 
- What steps did you take in your attempts to increase model performance?
 - Lasso and Ridge tests to drop low impact columns
 - PCA the data to reduce scaled data columns
 - Alter the activation functions, layer counts, and neuron counts per layer via Keras Tuner

###### Summary: 
In result, with a target accuracy of 75% the neural model reached an accuracy of 75.5%. I do believe different models such as, RandomForest might result with a better predictive accuracy. Using this dataset, it goes to show that a neural network that have processing being hidden it might be difficult to show the corrilation between two columns or a feature column to the target column.
