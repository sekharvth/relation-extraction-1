# import the necessary packages
import numpy as np
from keras.models import Model
from keras.layers import Lambda, Input, Embedding, Concatenate, Conv1D, MaxPooling1D, Flatten, Reshape, Dense
import keras.backend as K

# read in the 50 dimensional GloVe vectors from the file 
def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
    return word_to_vec_map

word_to_vec_map = read_glove_vecs('/data/file')

# make a temporary matrix that will store the embeddings of the words in our dictionary, 'word_to_index'.
# here, vector_dims = 50
embedding_matrix = np.zeros((vocab_size, vector_dims))

# fill in the embedding matrix with the embeddings of each word, corresponding to its index in our dictionary
for word,index in word_to_index.items():
    try:
        embedding_matrix[index, :] = word_to_vec_map[word.lower()]
    except:
        embedding_matrix[index, :] = np.random.uniform(0, 1, vector_dims)

# make a Keras embedding layer that is initialized with 50d embeddings of each word, but make it trainable so that 
# word vectors can be updated to contexts in our specific data set
embed_layer = Embedding(vocab_size, vector_dims, trainable = True)
embed_layer.build((None,))
embed_layer.set_weights([embedding_matrix])

# here I assume that the maximum length of each sentence is 20 
# make a Keras embedding layer for each position matrix (one for each word in an entity pair), and use Xavier initialization
position_layer_1 = Embedding(maxlen_of_sentences, pos_dims, initializer = 'glorot_uniform', seed = 0)
position_layer_2 = Embedding(maxlen_of_sentences, pos_dims, initializer = 'glorot_uniform', seed = 0)

# define the two inputs to the model.
# x1 is the concatenation of all sentences for a given entity pair, padded to make all entity pairs have the same number 
# of sentences. Its actual shape is (num_entity_pairs, max_num_of_sentences_for_an_entity_pair). 
# x2 is the concatenation of each position vector for each sentence. For each sentence, the entity pair contains 2 words,
# and thus there will be 2 postion vectors for each sentence. All these are concatenated, and so the number of entries will be 
# twice the number of sentences.
x1 = Input(shape = (max_num_of_sentences_for_an_entity_pair, ))
x2 = Input(shape = (2 * max_num_of_sentences_for_an_entity_pair, ))

# an empty list to store the outputs.
outputs = []
# loop through the concatenated sentences and select each sentence and its corresponing position vectors
for i in range(int(max_num_of_sentences_for_an_entity_pair/maxlen_of_sentences)):
    x1 = Lambda(lambda x: x[:, (i * maxlen_of_sentences) : ((i+1) * maxlen_of_sentences)])(x1)
    x2 = Lambda(lambda x: x[:, (2 * i * maxlen_of_sentences) : (2 * (i+1) * maxlen_of_sentences)])(x2)

    x2_pos_matrix_1 = Lambda(lambda x: x[:, :maxlen_of_sentences])(x2)
    x2_pos_matrix_2 = Lambda(lambda x: x[:, maxlen_of_setences:])(x2)
    
    # pass each input into the embedding layer to generate embeddings
    input_embed = embed_layer(x1)
    pos_embed_1 = position_layer_1(x2_pos_matrix_1)
    pos_embed_2 = position_layer_2(x2_pos_matrix_2)

    # now concatenate all the embeddings for each word in each sentence
    merged = Concatenate(axis = -1)([input_embed, pos_embed_1, pos_embed_2])
    
    # pass it through a 1D convolution, with 100 filters and relu activation. 
    # Resulting shape is (20(maxlen) - 3(kernel_size) + 1, 100(num_filters))
    conv = Conv1D(filters = 100, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(merged)
    
    # perform piece-wise max pooling. Resulting shape is (6, 100)
    piece_pool = MaxPooling1D(pool_size = 3, strides = 3, padding = 'same')(conv)
    
    # flatten the outputs and make it into the desired shape
    piece_pool = Flatten()(piece_pool)
    piece_pool = Reshape((600, 1))(piece_pool)
    
    # append the outputs of each sentence to 'outputs' list
    outputs.append(piece_pool)

# concatenate the outputs of all the sentences
concat = Concatenate(axis = -1)(outputs)

# Take the maximum of each feature across all sentences
cross_sentence_pool = Lambda(lambda x: K.max(x, axis = -1, keepdims = True))(concat)

# add dropout for regularization
cross_sentence_pool_with_dropout = Dropout(0.5)(cross_sentence_pool)

# flatten the output and pass it through a Dense layer, having 'num_of_classes' units, and sigmoid activation for multi-label classification
flat = Flatten()(cross_sentence_pool_with_dropout)
results = Dense(num_relations, activation = 'sigmoid')(flat)
    
# create the Model instance with the inputs of x1 and x2, and outputs the probability of each class, and compile it
model = Model([x1, x2], results)
model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
                         
# fit the model
model.fit([all_entity_pair_sentences, all_position_matrices], all_relations_for_each_pair, epochs = 100, batch_size = 64)

# fit the model
model.fit([all_entity_pair_sentences, all_position_matrices], all_relations_for_each_pair, epochs = 100, batch_size = 64)
