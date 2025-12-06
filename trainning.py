import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import os

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents JSON
data_file = open('data.json').read()
intents = json.loads(data_file)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process each pattern and intent
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents
        documents.append((w, intent['tag']))
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lowercase, remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")

# Save words and classes
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    # Bag of words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    bag = [1 if w in pattern_words else 0 for w in words]

    # Output row
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle and convert to NumPy arrays
random.shuffle(training)
training = np.array(training, dtype=object)  # use dtype=object to avoid inhomogeneous errors

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

print("Training data created")

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model
model.save('model.h5')
print("Model created and saved as model.h5")
