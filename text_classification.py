import tensorflow as td
from tensorflow import keras
import numpy as np


data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
# print(train_data[1]," " ,train_labels[1]) #train_labels => either 0 or 1, 0=negative ,1=positive review

word_index = data.get_word_index()
word_index={k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0 # used for making all reviews into same length
word_index["<START>"]=1
word_index["<UNK>"]=2
word_index["<UNUSED>"]=3
#interchanging values as keys.
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
#trim down each review to 250 characters
train_data=keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",  maxlen=250)
test_data=keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",  maxlen=250)



def decode_review(text): #function called when we display result as words
    return " ".join([reverse_word_index.get(i,"?") for i in text])
# print(decode_review(train_data[0]))
#print(decode_review(test_data[2]))
#model below
#
# model = keras.Sequential()
# model.add(keras.layers.Embedding(880000,16)) #16 dimensions , every data is a vector
# model.add(keras.layers.GlobalAveragePooling1D()) # trim down the 16D
# model.add(keras.layers.Dense(16,activation="relu")) # 16 neurons for classification
# model.add(keras.layers.Dense(1,activation="sigmoid")) # sigmoid squeeze between 0 and 1, compare with softmax for getting the idea
# #embedding layer -> generates word vector and forms group with context
#
# model.summary()
# model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"]) # calculates loss from test
#
# x_val = train_data[:10000]
# x_train = train_data[10000:]
#
# y_val = train_labels[:10000]
# y_train = train_labels[10000:]
#
# fitModel = model.fit(x_train,y_train,epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)
#
# results = model.evaluate(test_data, test_labels)
#
# print(results)
# model.save("model.h5") #h5 extension

#model ends here

def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return  encoded
model = keras.models.load_model("model.h5");

# with open("lion_king_review.txt",encoding="utf-8") as f:
def find(review):
    print("works")
    review = review.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","").strip().split(" ")
    encode = review_encode(review)
    encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",  maxlen=250)
    predict = model.predict(encode)
    print(line)
    print(encode)
    print(predict[0])
    return predict[0]


# test_review = test_data[1]
# predict = model.predict([test_review])
# print("Review : ")
# print(decode_review(test_review))
# print("Prediction : " + str(predict[1]))
# print("Actual : " + str(test_labels[1]))

