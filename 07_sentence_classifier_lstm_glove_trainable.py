import os
import sys
import re
from time import time
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Embedding, Dropout, Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder


class IntentClassifer:
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 51
        self.EMBEDDING_DIM = 100
        self.EMBED_DIR = './embeddings'
        self.WORD_INDEX = dict()
        self.LABEL_ENCODER = None
        self.LABEL_COUNT = None
        self.VALIDATION_SPLIT = 0.0

    @staticmethod
    def read_data(training_data_df_):
        x_series = training_data_df_["Query"].tolist()
        y_ = training_data_df_["Type"].tolist()
        return x_series, y_

    @staticmethod
    def load_test(test_data_df_):
        x_series = test_data_df_["Query"].tolist()
        y_ = test_data_df_["Type"].tolist()
        return x_series, y_

    @staticmethod
    def nlp_layer(x_series_):
        contractions_list = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'd've": "i would have",
            "i'll": "i will",
            "i'll've": "i will have",
            "i'm": "i am",
            "i'am": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it had",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there had",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we had",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'alls": "you alls",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you had",
            "you'd've": "you would have",
            "you'll": "you you will",
            "you'll've": "you you will have",
            "you're": "you are",
            "you've": "you have"
        }
        c_re_ = re.compile('(%s)' % '|'.join(contractions_list.keys()))

        def expand_contractions(text, c_re=c_re_):
            def replace(match):
                return contractions_list[match.group(0)]

            return c_re.sub(replace, text)

        x_ = [re.sub('[^0-9a-z\' ]+', ' ', item.lower()).split() for item in x_series_]

        processed_x = []
        for eachQuery in x_:
            query = []
            for eachToken in eachQuery:
                expanded_token = expand_contractions(eachToken)
                # Applied to normalize words like word's, mother's, 1980's etc.
                if '\'' in expanded_token:
                    expanded_token = expanded_token.strip('\'s')
                # Replace alpha-numeric with special token 'Client_Tok'
                if expanded_token.isalnum() and not expanded_token.isalpha() and not expanded_token.isdigit():
                    expanded_token = 'ClientTok'
                if expanded_token.isdigit():
                    expanded_token = 'NUM'
                query.extend(expanded_token.split())
            processed_x.append(query)
        return processed_x

    def integer_encode(self, x_train_, y_train_):
        tokenizer = Tokenizer(oov_token='UNK')
        tokenizer.fit_on_texts(x_train_)
        self.WORD_INDEX = tokenizer.word_index
        x_train_encoded = tokenizer.texts_to_sequences(x_train_)
        x_train_encoded_padded = pad_sequences(x_train_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='pre')

        encoder = LabelEncoder()
        encoder.fit(y_train_)
        self.LABEL_ENCODER = encoder
        y_train_encoded = encoder.transform(y_train_)
        y_train_one_hot = np_utils.to_categorical(y_train_encoded)
        self.LABEL_COUNT = y_train_one_hot.shape[1]
        print("\tUnique Tokens in Training Data: %s" % len(self.WORD_INDEX))
        print("\tNo. of classes: %s" % self.LABEL_COUNT)
        print("\tShape of data tensor (X_train): %s" % str(x_train_encoded_padded.shape))
        print("\tShape of label tensor (Y): %s" % str(y_train_one_hot.shape))
        return x_train_encoded_padded, y_train_one_hot

    def integer_encode_test(self, x_test_, y_test_):
        x_test_encoded = list()
        for sentence in x_test_:
            x_test_ = [self.WORD_INDEX[w] for w in sentence if w in self.WORD_INDEX]
            x_test_encoded.append(x_test_)
        x_test_encoded_padded = pad_sequences(x_test_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='pre')

        y_test_encoded = self.LABEL_ENCODER.transform(y_test_)
        y_test_one_hot = np_utils.to_categorical(y_test_encoded, num_classes=len(self.LABEL_ENCODER.classes_))
        print("\tUnique Tokens in Test Data (this should be same as in Training Data): %s" % len(self.WORD_INDEX))
        print("\tShape of data tensor (X_test): %s" % str(x_test_encoded_padded.shape))
        print("\tShape of label tensor (Y_test): %s" % str(y_test_one_hot.shape))
        return x_test_encoded_padded, y_test_one_hot

    def load_glove(self):
        embeddings_index_ = {}
        try:
            f = open(os.path.join(self.EMBED_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
        except FileNotFoundError:
            print("GloVe vectors missing. You can download from http://nlp.stanford.edu/data/glove.6B.zip")
            sys.exit()
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index_[word] = coefs
        f.close()
        print("\tNumber of Tokens from GloVe: %s" % len(embeddings_index_))
        return embeddings_index_

    def glove_embedding_matrix(self, embeddings_index_):
        words_not_found = []
        vocab = len(self.WORD_INDEX) + 1
        # embedding_matrix = np.zeros((vocab, self.EMBEDDING_DIM))
        # 0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        embedding_matrix_ = np.random.uniform(-0.25, 0.25, size=(vocab, self.EMBEDDING_DIM))
        for word, i_ in self.WORD_INDEX.items():
            if i_ >= vocab:
                continue
            embedding_vector = embeddings_index_.get(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                embedding_matrix_[i_] = embedding_vector
            else:
                words_not_found.append(word)
        # print('Number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        print("\tShape of embedding matrix: %s" % str(embedding_matrix_.shape))
        print("\tNo. of words not found in GloVe: ", len(words_not_found))
        print("Words not found:", words_not_found)
        return embedding_matrix_

    def lstm_model(self, embedding_matrix_):
        model_ = Sequential()
        model_.add(Embedding(len(self.WORD_INDEX) + 1, self.EMBEDDING_DIM, weights=[embedding_matrix_],
                             input_length=self.MAX_SEQUENCE_LENGTH, trainable=True))
        model_.add(LSTM(512))
        model_.add(Dense(1024, activation='relu'))
        model_.add(Dropout(0.2))
        model_.add(Dense(self.LABEL_COUNT, activation='softmax'))
        adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model_.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
        # print("Model Summary")
        # print(model_.summary())
        hist = model_.fit(x_train, y_train,
                          batch_size=64,
                          shuffle=True,
                          epochs=25,
                          verbose=2)
        return model_, hist.history['acc'][-1]


if __name__ == '__main__':

    training_data = "./data-set/trec10_train.csv"
    test_data = "./data-set/trec10_test.csv"

    print('Training data received from user: %s' % training_data)
    training_data_df = pd.read_csv(training_data, encoding='utf-8')

    sampleInstance = IntentClassifer()
    x_series_train, y = sampleInstance.read_data(training_data_df)
    x = sampleInstance.nlp_layer(x_series_train)
    print("X_train(samples):", x[:2])
    print("Y_train(samples):", y[:2])
    X_encoded_padded, Y_one_hot = sampleInstance.integer_encode(x, y)
    x_train, y_train = X_encoded_padded, Y_one_hot
    embeddings_index = sampleInstance.load_glove()
    times = list()
    train_accuracies = list()
    test_accuracies = list()
    epoch = 5
    for i in range(epoch):
        print("*" * 25)
        print("Run %s/%s" % (i + 1, epoch))
        embedding_matrix = sampleInstance.glove_embedding_matrix(embeddings_index)
        train_start_time = time()
        model, train_acc = sampleInstance.lstm_model(embedding_matrix)
        print("Training accuracy: ", train_acc)
        train_end_time = time()
        time_taken = round((train_end_time - train_start_time), 2)
        times.append(time_taken)
        train_accuracies.append(round(train_acc, 2))
        print("Training time taken is %s units." % (str(time_taken)))

        # Testing the model
        print("Evaluating the model.")
        print('Test data received from user: %s' % test_data)
        test_data_df = pd.read_csv(test_data, encoding='utf-8')
        x_series_test, Y_test = sampleInstance.load_test(test_data_df)
        x_test = sampleInstance.nlp_layer(x_series_test)
        print("X_Test(samples):", x_test[:2])
        print("Y_Test(samples):", Y_test[:2])
        X_test_encoded_padded, Y_test_one_hot = sampleInstance.integer_encode_test(x_test, Y_test)
        scores = model.evaluate(X_test_encoded_padded, Y_test_one_hot, verbose=0)
        print("Test %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        test_accuracies.append(round(scores[1], 2))

    print("Training Times(in Sec.): ")
    print(times)
    print("Training Accuracies:")
    print(train_accuracies)
    print("Test Accuracies: ")
    print(test_accuracies)

