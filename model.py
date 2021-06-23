import tensorflow as tf
from tensorflow.keras import Sequential, Dense, layers, partial


class Model:
    MAX_EPOCHS = 10
    multi_val_performance = {}
    multi_performance = {}
    num_label=2
    hiddennodes=40
    OUT_STEPS = 1

    def generate_model(self):
        model = Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            # tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(hiddennodes, activation=partial(tf.nn.leaky_relu, alpha=0.5)),
            tf.keras.layers.Dense(OUT_STEPS*num_label,
                         kernel_initializer=tf.initializers.zeros
                         ),
        ])

    def compile(self, model, lr=0.001):
        model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=lr),
                metrics=[tf.metrics.MeanSquaredError()],
                   experimental_steps_per_execution=10)

    def fit(self, model, window, patience=150):
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
            # callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
            history = model.fit(window.train, epochs=MAX_EPOCHS,
                     validation_data=window.val,
                  #    callbacks=[early_stopping,
                   #  callback
                         #       ]
                     )
            return history