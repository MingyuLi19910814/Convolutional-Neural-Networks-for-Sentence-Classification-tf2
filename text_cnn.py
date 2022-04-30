import tensorflow as tf

def create_text_cnn(sequence_length,
                    num_classes,
                    vocab_size,
                    embedding_size,
                    filter_sizes,
                    num_filters,
                    dropout_keep_prob,
                    l2_reg_lambda=0.0):
    input_x = tf.keras.layers.Input(shape=(sequence_length,), dtype=tf.int32)  # (batch_size, sequence_length)
    embedded_x = tf.keras.layers.Embedding(input_dim=vocab_size,
                                           output_dim=embedding_size,
                                           embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1))(input_x)
    embedded_x = tf.expand_dims(embedded_x, -1) # (batch_size, sequence_length, embedding_size, 1)
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        x = tf.keras.layers.Conv2D(filters=num_filters,
                                   kernel_size=(filter_size, embedding_size),
                                   strides=(1, 1),
                                   padding='VALID',
                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                   bias_initializer=tf.keras.initializers.Constant(0.1),
                                   activation='relu')(embedded_x) # (batch_size, sequence_length - filter_size + 1, 1, num_filters)
        x = tf.keras.layers.MaxPooling2D(pool_size=(sequence_length - filter_size + 1, 1),
                                         strides=(1, 1),
                                         padding='VALID')(x) #(batch_size, 1, 1, num_filters)
        pooled_outputs.append(x)
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, axis=3) #(batch_size, 1, 1, num_filters_total)
    h_pool_flat = tf.reshape(h_pool, (-1, num_filters_total)) #(batch_size, num_filters_total)
    h_drop = tf.keras.layers.Dropout(rate=dropout_keep_prob)(h_pool_flat) #(batch_size, num_filters_total)
    output = tf.keras.layers.Dense(num_classes,
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                   bias_initializer=tf.keras.initializers.Constant(0.1))(h_drop)
    model = tf.keras.Model(inputs=input_x, outputs=output)
    return model