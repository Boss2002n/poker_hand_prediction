# %%
import tensorflow as tf
import matplotlib.pyplot as plt

# %% [markdown]
#  **Introduction**
#  
#  My model that aims to learn complex patterns in the input data to accurately classify different poker hands. 
#  
#  By adjusting the model architecture, parameters and regularization techniques- the goal is to improve the model performance and achieve higher accuracy in predicting poker hands
# %%
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
poker_hand = fetch_ucirepo(id=158) 

# data (as pandas dataframes) 
X = poker_hand.data.features 
y = poker_hand.data.targets 

# metadata 
print(poker_hand.metadata) 
  
# variable information 
print(poker_hand.variables) 


# %% [markdown]
# **Objective**
# 
# 
# The primary objective is to construct a neural network model capable of effectively categorizing poker hands into one of ten classes, ranging from "Nothing in hand" to "Royal flush." By leveraging techniques such as data preprocessing, feature engineering, and neural network architecture design, the aim is to achieve high accuracy and robust performance on unseen data.

# %%
# Define hand descriptions
hand_descriptions = {
    0: "Nothing in hand; not a recognized poker hand",
    1: "One pair; one pair of equal ranks within five cards",
    2: "Two pairs; two pairs of equal ranks within five cards",
    3: "Three of a kind; three equal ranks within five cards",
    4: "Straight; five cards, sequentially ranked with no gaps",
    5: "Flush; five cards with the same suit",
    6: "Full house; pair + different rank three of a kind",
    7: "Four of a kind; four equal ranks within five cards",
    8: "Straight flush; straight + flush",
    9: "Royal flush; {Ace, King, Queen, Jack, Ten} + flush"
}

# %% [markdown]
# Split the dataset into features (X) and the target variable (y).
# 
# Convert categorical features (S1, S2, S3, S4, S5) into one-hot encoded vectors.
# 
# Normalize numerical features (C1, C2, C3, C4, C5) to have zero mean and unit variance.
# 
# Split the dataset into training and testing sets

# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Select features
numerical_features = ['C1', 'C2', 'C3', 'C4', 'C5']
categorical_features = ['S1', 'S2', 'S3', 'S4', 'S5']

# Normalize numerical features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X[numerical_features])

# Combine numerical features
X_processed = np.concatenate((X_numerical, X[categorical_features]), axis=1)

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Print shapes of training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# %% [markdown]
# The following will help you understand the data better
# 
# 1) S1 "Suit of card #1"
#     Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
# 
# 2) C1 "Rank of card #1"
#     Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
# 
# 3) S2 "Suit of card #2"
#     Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
# 
# 4) C2 "Rank of card #2"
#     Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
# 
# 5) S3 "Suit of card #3"
#     Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
# 
# 6) C3 "Rank of card #3"
#     Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
# 
# 7) S4 "Suit of card #4"
#     Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
# 
# 8) C4 "Rank of card #4"
#     Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
# 
# 9) S5 "Suit of card #5"
#     Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
# 
# 10) C5 "Rank of card 5"
#     Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
# 
# 11) CLASS "Poker Hand"
#     Ordinal (0-9)
# 
#     0: Nothing in hand; not a recognized poker hand 
# 
#     1: One pair; one pair of equal ranks within five cards
# 
#     2: Two pairs; two pairs of equal ranks within five cards
# 
#     3: Three of a kind; three equal ranks within five cards
# 
#     4: Straight; five cards, sequentially ranked with no gaps
# 
#     5: Flush; five cards with the same suit
# 
#     6: Full house; pair + different rank three of a kind
# 
#     7: Four of a kind; four equal ranks within five cards
# 
#     8: Straight flush; straight + flush
#     
#     9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush

# %% [markdown]
# Below a sequential model is initialized that represents a bunch of layers.
# 
# The reason I went with a sequential model vs a functional model was that the sequential model is simpler and leaves less scope of error[1]https://www.youtube.com/watch?v=EvGS3VAsG4Y
# 
# other hidden layers are added sequentially to the model, which include an input layer with 256 neurons and relu activation, 
# followed by three hidden layers with 128 and 64 neurons using relu activation. 
# 
# Finally, an output layer with 10 neurons and softmax activation is added.
# 
# Adding more layers has both advantages and disadvantages- 
# 
# Adding more layer would make the model develop intricate dependancies, and would hep the model to generalize new and unseen examples but that comes with a risk of overfitting* and would be resource hungry - increased training time and computational time (not very practical)
# 
# 
# 
# *the model would learn all the noise in the dataset - similar to hardcoding - a dropoutlayer would help to ensure that overfitting doesnt happen
# 
# A droput layer randomly deactivates a neuron with the specified probability. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout#:~:text=The%20Dropout%20layer%20randomly%20sets,over%20all%20inputs%20is%20unchanged

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Define the neural network model
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))  # Input layer
model.add(Dropout(0.4))  # Dropout layer to prevent overfitting 
model.add(Dense(128, activation='relu'))  # First hidden layer
model.add(Dense(64, activation='relu'))  # Second hidden layer
model.add(Dense(10, activation='softmax'))  # Output layer



# %% [markdown]
# hidden layers play a crucial role in enabling neural networks to learn and generalize from complex data by extracting hierarchical features, introducing non-linearities, reducing dimensionality, and increasing the model's capacity to capture intricate patterns. [2]https://www.youtube.com/watch?v=Y1qxI-Df4Lk

# %% [markdown]
# Below we initialize the optimizer with a learning rate of 0.001, compile the model with a specified metric which is to be monitored during the training process (accuraccy in this case)
# 
# The loss function used is sparse categorical crossentropy (scce) because my classes are mutually exclusive. [3]
# 
# scce produces a category index of the most likely matching category.
# 
# https://fmorenovr.medium.com/sparse-categorical-cross-entropy-vs-categorical-cross-entropy-ea01d0392d28

# %%
from keras.optimizers import Adam

# Compile the model
adam = Adam(learning_rate=0.001) 
model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# Print model summary
model.summary()

# %% [markdown]
# Now we will train teh model using the training data X_train and y_train, the epochs parameter is set to 14 therefore the model will be trained for 14 times with the training dataset
# 
# Training is performed in batches using a subset of the training data is processed. Here we chose a batch size of 256.
# 
# A portion of the training data is set aside for validation using the validation split parameter, in this case it is set to 20% (0.2), adter each run/epoch the model is evaluated based on the dataset which was set aside.
# 
# The smaller the batch size, the model will observe more intricate details and would be better in terms of accuracy. The downside for using small batch sizes is that the time taken to train the model would be very high leading to a slow model. 
# 
# I have targeted to reach 95% accuracy which is pretty high

# %%
# Fit the model to the training data
history = model.fit(X_train, y_train, epochs=14, batch_size=256, validation_split=0.2)


# %% [markdown]
# Below I visually depict the accuracy and loss history to track the performance of the model

# %%
# Plot accuracy and loss history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epoch')
plt.title('Improved Model Performance')
plt.legend()
plt.show()

# %% [markdown]
# Predicted class labels are obtained by selecting the index of the maximum probability 
# 
# Subsequently, the integer-encoded predicted and true labels are decoded back to their original class labels using a label encoder 
# 
# This helps the humans to read the models predictions in human readable format.

# %%
from sklearn.metrics import classification_report

# Evaluating the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

print("Testing Accuracy:", accuracy)

# predicting classes
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Decode integer labels back to original class labels
# https://stackoverflow.com/questions/52870022/inverse-transform-method-labelencoder
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred_labels)

# Print classification report
print(classification_report(y_test_labels, y_pred_labels))


# %% [markdown]
# The classification report shows a representation of the main classification metrics on a per-class basis
# 
# precision is the an indication of the fraction of your predictions were correct
# 
# recall is the an indication of the fraction of the positive cases did you catch
# 
# f1-score is the an indication of the fraction of positive predictions were correct
# 
# Support is the number of actual occurrences of the class in the specified dataset
# 
# The accuracy for class 9 and 8 would be lower because the occurrences for this event in the data set are very less (2,3 resp.)

# %% [markdown]
# Now we test our models predictions.
# 
# I have explained how to understand a sample below in a markdown cell.

# %%
# THis function refines the card numbers in the sample, it replaces the face card numbers to a string which represents the face card.
def refine_card_numbers(sample):
    # Retrieving the first 5 entires in the sample which represent the numbers
    standardized_data = np.array([sample[0][:5]])

    # Decoding the entries.
    refined_numbers = scaler.inverse_transform(standardized_data).astype(int).tolist()
    refined_numbers = refined_numbers[0] 

    # Replacing the face card numbers with strings.
    for i, num in enumerate(refined_numbers):
        if num == 1:
            refined_numbers[i] = "Ace"
        elif num == 11:
            refined_numbers[i] = "Jack"
        elif num == 12:
            refined_numbers[i] = "Queen"
        elif num == 13:
            refined_numbers[i] = "King"
    return refined_numbers

# This function returns the refines suits for better readability.
def refine_suits(sample):
    # Hash map of the suits and their representative numbers
    suit_mapping = {1: 'Hearts', 2: 'Spades', 3: 'Diamonds', 4: 'Clubs'}
    # Retrieving the last 5 entires in the sample which represent the suits
    standardized_data = np.array([sample[0][5:]])
    # Converting to list and integers for easier data manipulation
    original_suits = standardized_data.astype(int).tolist()
    # Iterating through the list and replacing the numbers wiht the respective strings.
    refined_suits = [suit_mapping[suit] for suit in original_suits[0]]
    return refined_suits

# Getting an input as to how many samples we want
num_runs = int(input("How many samples you want to run?"))

for _ in range(num_runs):
    # selecting a random sample from the test dataset
    import random
    random_index = random.randint(0, len(X_test) - 1)
    sample = X_test[random_index]
    sample_label = y_test_labels[random_index]

    # matching the input shape of the model
    sample = sample.reshape(1, -1)

    # here's where the magic takes place - predicting the class probabilities
    predicted_probabilities = model.predict(sample)[0]

    # picking the highest probability in the possibilities
    predicted_label_index = np.argmax(predicted_probabilities)

    # Retrieving the class label which is at index [0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    
    # Getting our prediction into human-understanable format
    predicted_hand_description = hand_descriptions[predicted_label_index]

    # printing the sample and our models prediction
    print("Raw Sample:", sample)
    
    print("\n\nRefined Sample Numbers:", refine_card_numbers(sample))
    print("Refined Sample Suits:", refine_suits(sample))
    print("Actual Label:", sample_label)
    print("Predicted Label:", predicted_label)
    print("Predicted Hand Description:", predicted_hand_description)
    print("-"*100)



# %% [markdown]
# Example sample and how to understand it - 
# 
# [[ 0.5348267  -0.80295204  1.33639879 -1.06933645  0.80514033 
# 
#     4.         1.          2.          2.          2.        ]]
# 
# 
# Based on the attributes in the sample, it represents the following:
# 
# 0.5348267: This represents a rank of approximately 9 (since the rank values are standardized, the exact value may not correspond directly to a specific rank).
# 
# -0.80295204: This represents a rank of approximately 3
# 
# 1.33639879: This represents a rank of approximately 12 (Queen)
# 
# -1.06933645: This represents a rank of approximately 2
# 
# 0.80514033: This represents a rank of approximately 10
# 
# These numerical values are standardized, meaning they've been transformed to have a mean of 0 and a standard deviation of 1. Therefore, the exact numerical values don't directly correspond to the traditional ranks, but they can be interpreted relative to each other.
# 
# C1: 9
# 
# C2: 3
# 
# C3: 12 (or Queen)
# 
# C4: 2
# 
# C5: 10
# 
# 
# 
# 4.0: This represents the suit of the first card in the hand. In this dataset, the suits are represented as ordinal values ranging from 1 to 4.
# 
# 1.0: This represents the suit of the second card in the hand.
# 
# 2.0: This represents the suit of the third card in the hand.
# 
# 2.0: This represents the suit of the fourth card in the hand.
# 
# 2.0: This represents the suit of the fifth card in the hand.
# 
# 
# S1: Clubs
# 
# S2: Hearts
# 
# S3: Spades
# 
# S4: Spades
# 
# S5: Spades

# %%
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices())



