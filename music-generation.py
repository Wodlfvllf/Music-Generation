import numpy as np
import tensorflow as tf
import soundfile as sf
import os
from tqdm import tqdm

class LSTM(tf.keras.Model):
    def __init__(self, lstm_units, vocab_size, embedding_dim, batch_size):
        super(LSTM, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None])
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, recurrent_initializer="glorot_uniform", recurrent_activation='sigmoid', stateful=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        return x

class MusicGenerator:
    def __init__(self, num_training_iterations, batch_size, seq_length, learning_rate, checkpoint_dir, rnn_units, vocab_size, embedding_dim):
        self.num_training_iterations = num_training_iterations
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.losses = []
        self.rnn_units = rnn_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
        self.model = LSTM(rnn_units, vocab_size, embedding_dim, batch_size)  # Create an instance of the LSTM model

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = compute_loss(y, y_pred)
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return loss.numpy().mean()

    def train(self, vectorized_songs):
        for i in tqdm(range(0, self.num_training_iterations)):
            x, y = get_batch(vectorized_songs, self.seq_length, self.batch_size)
            loss = self.train_step(x, y)
            self.losses.append(loss)

            if i % 100 == 0:
                self.model.save_weights(self.checkpoint_prefix)

        self.model.save_weights(self.checkpoint_prefix)

    def generate_and_save(self, generated_songs):
        output_directory = 'output_music/'
        os.makedirs(output_directory, exist_ok=True)
        for i, song in enumerate(generated_songs):
            waveform = mdl.lab1.play_song(song)
            if waveform:
                filename = os.path.join(output_directory, f"generated_song_{i}.wav")
                sf.write(filename, waveform, mdl.lab1.sample_rate)

def main():
    num_training_iterations = 5000  # Increase this to train longer
    batch_size = 4  # Experiment between 1 and 64
    seq_length = 100  # Experiment between 50 and 500
    learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1
    checkpoint_dir = './Waveform'
    checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

    # Model parameters:
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024

    music_generator = MusicGenerator(num_training_iterations, batch_size, seq_length, learning_rate, checkpoint_dir, rnn_units, vocab_size, embedding_dim)
    music_generator.train(vectorized_songs)

    music_generator.model = LSTM(rnn_units, vocab_size, embedding_dim, batch_size)
    music_generator.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    music_generator.model.build(tf.TensorShape([1, None]))

    generated_text = generate_text(music_generator.model, start_string="X", generation_length=10000)
    generated_songs = mdl.lab1.extract_song_snippet(generated_text)

    music_generator.generate_and_save(generated_songs)

if __name__ == "__main__":
    main()
