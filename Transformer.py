import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import datetime
import warnings
import gc
import os, sys

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['PYTHONHASHSEED']=str(0)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

K.clear_session()
tf.compat.v1.reset_default_graph()
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()
np.random.seed(0)

def function_1():
	keys = ['Time', 'Price']
	dataset_org = pd.read_csv('YOUR_DATASET_HERE', index_col=False, names=keys, header=None, dtype=object, error_bad_lines=False, warn_bad_lines=False, skipinitialspace=True, usecols=keys).apply(pd.to_numeric, errors = 'coerce')
	for i_pd in range(0, len(dataset_org)+1):
		try:
			dataset = dataset_org.iloc[:i_pd,:].tail(100).dropna(axis=1, how='all').fillna(0).round(5)
			dataset = dataset.drop_duplicates(subset=['Time'])
			dataset = dataset.reset_index(drop=True)

			dataset['Time'] = dataset['Time'].apply(lambda x: x/1000000000)
			current_time = float(dataset['Time'].iloc[-1])
			current_price = float(dataset.Price.iloc[-1])

			dataset['Price'] = dataset.Price.pct_change().fillna(0)
			dataset['final_y'] = dataset.Price.shift(-1).fillna(0)
			dataset = dataset[['Price', 'final_y']]

			dataset = dataset.loc[:, (dataset != 0).any(axis=0)]
			dataset = dataset.replace([np.inf, -np.inf], np.nan).fillna(0)
			rows_1 = len(dataset)-1

			dataset = dataset.to_numpy()

			dataset = np.multiply(dataset, 10000000).astype(int)
			negative_0 = int(np.abs(np.amin(dataset)))
			dataset = np.add(dataset, negative_0)
			negative_1 = int(np.abs(np.amax(dataset)))
			dataset = np.add(dataset, negative_1)

			reward = dataset[:-1,-1:]
			state = dataset[:-1,:-1]
			to_Predict = dataset[-rows_1:,:-1]
			to_Predict = to_Predict.astype(int).reshape((rows_1, 1))
			state = state.astype(int).reshape((rows_1, 1))
			reward = reward.astype(int).reshape((rows_1, 1))
			vocab_size = np.unique(reward).size
			vocab_max = np.amax(dataset.flatten())
			print(reward.flatten())

			d_model_1 = rows_1
			if d_model_1 % 2 != 0:
				d_model_1 = d_model_1+1

			def positional_encoding(length, depth):
				depth = depth/2

				positions = np.arange(length)[:, np.newaxis]	 # (seq, 1)
				depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

				angle_rates = 1 / (100**depths)	   # (1, depth)
				angle_rads = positions * angle_rates	  # (pos, depth)

				pos_encoding = np.concatenate(
					[np.sin(angle_rads), np.cos(angle_rads)],
					axis=-1) 

				return tf.cast(pos_encoding, dtype=tf.float32)

			class PositionalEmbedding(tf.keras.layers.Layer):
				def __init__(self, vocab_size, d_model):
					super().__init__()
					self.d_model = d_model
					self.embedding = tf.keras.layers.Embedding(vocab_max+1, d_model_1, mask_zero=True) 
					self.pos_encoding = positional_encoding(length=d_model_1*4, depth=d_model_1)

				def compute_mask(self, *args, **kwargs):
					return self.embedding.compute_mask(*args, **kwargs)

				def call(self, x):
					length = tf.shape(x)[1]
					x = self.embedding(x)
					x *= tf.math.sqrt(tf.cast(rows_1, tf.float32))
					x = x + self.pos_encoding[tf.newaxis, :length, :]
					return x


			class BaseAttention(tf.keras.layers.Layer):
				def __init__(self, **kwargs):
					super().__init__()
					self.mha = tf.keras.layers.MultiHeadAttention(**kwargs, use_bias=True, trainable=True)
					self.layernorm = tf.keras.layers.LayerNormalization()
					self.add = tf.keras.layers.Add()

			class CausalSelfAttention(BaseAttention):
				def call(self, x):
					attn_output = self.mha(
						query=x,
						value=x,
						key=x,
						use_causal_mask = True)
					x = self.add([x, attn_output])
					return x

			class FeedForward(tf.keras.layers.Layer):
				def __init__(self, d_model, dff, dropout_rate=0.1):
					super().__init__()
					self.seq = tf.keras.Sequential([
					tf.keras.layers.Dense(dff, activation='relu'),
					tf.keras.layers.Dense(d_model),
					tf.keras.layers.Dropout(dropout_rate)
					])
					self.add = tf.keras.layers.Add()
					self.layer_norm = tf.keras.layers.LayerNormalization()

				def call(self, x):
					x = self.add([x, self.seq(x)])
					x = self.layer_norm(x) 
					return x

			class DecoderLayer(tf.keras.layers.Layer):
				def __init__(self,
							*,
							d_model,
							num_heads,
							dff,
							dropout_rate=0.1):
					super(DecoderLayer, self).__init__()

					self.causal_self_attention = CausalSelfAttention(
						num_heads=num_heads,
						key_dim=d_model,
						dropout=dropout_rate)

					self.ffn = FeedForward(d_model, dff)

				def call(self, x, context):
					x = self.causal_self_attention(x=x)
					x = self.cross_attention(x=x, context=context)

					# Cache the last attention scores for plotting later
					self.last_attn_scores = self.cross_attention.last_attn_scores

					x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
					return x

			class Decoder(tf.keras.layers.Layer):
				def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
							dropout_rate=0.1):
					super(Decoder, self).__init__()

					self.d_model = d_model
					self.num_layers = num_layers

					self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
															d_model=d_model)
					self.dropout = tf.keras.layers.Dropout(dropout_rate)
					self.dec_layers = [
						DecoderLayer(d_model=d_model, num_heads=num_heads,
									dff=dff, dropout_rate=dropout_rate)
						for _ in range(num_layers)]

					self.last_attn_scores = None

				def call(self, x, context):
					# `x` is token-IDs shape (batch, target_seq_len)
					x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

					x = self.dropout(x)

					for i in range(self.num_layers):
						x  = self.dec_layers[i](x, context)

					self.last_attn_scores = self.dec_layers[-1].last_attn_scores

					# The shape of x is (batch_size, target_seq_len, d_model).
					return x
					

			class Transformer(tf.keras.Model):
				def __init__(self, num_layers, d_model, num_heads, dff,target_vocab_size, dropout_rate):
					super().__init__()

					try:

						self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
											num_heads=num_heads, dff=dff,
											vocab_size=target_vocab_size,
											dropout_rate=dropout_rate)

						self.final_layer = tf.keras.layers.Dense(vocab_max+1, activation='softmax', use_bias=True, trainable=True)

					except Exception as e:
						exc_type, exc_obj, exc_tb = sys.exc_info()
						fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
						print(exc_type, fname, exc_tb.tb_lineno)
						print(e)

				def call(self, inputs, training=True):
					try:
						logits = tf.cast(inputs, tf.float32)
						logits = self.decoder(logits)
						logits = self.final_layer(logits)
						return logits
					except Exception as e:
						exc_type, exc_obj, exc_tb = sys.exc_info()
						fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
						print(exc_type, fname, exc_tb.tb_lineno)
						print(e)	

			num_layers = 1
			d_model = 1
			dff = 1
			num_heads = 1
			dropout_rate = 0

			transformer = Transformer(
				num_layers=num_layers,
				d_model=d_model,
				num_heads=num_heads,
				dff=dff,
				target_vocab_size=vocab_size,
				dropout_rate=dropout_rate)
			
			epoch_1 = 100
			earlyStopping = EarlyStopping(monitor='loss', mode='min', patience=epoch_1, restore_best_weights=True, verbose=0)

			transformer.compile(
				loss=tf.keras.losses.SparseCategoricalCrossentropy(),	
				optimizer=Adam(learning_rate=1/epoch_1))
			history = transformer.fit(x=state.reshape((rows_1, 1)).astype(int), y=reward.reshape((rows_1, 1)).astype(int), batch_size=rows_1, shuffle=False, epochs=epoch_1, verbose=0, callbacks=[earlyStopping])	
			hist_loss = history.history['loss']
			hist_loss_first = hist_loss[0]
			hist_loss_last = hist_loss[-1]
			to_Predict[np.isnan(to_Predict)] = 0
			to_predict_shape = tuple(list(to_Predict.shape))
			y_pred_org_1 = transformer.predict_step(to_Predict.reshape(to_predict_shape)).numpy()
			y_pred_org_1 = np.argmax(np.squeeze(y_pred_org_1, 1), -1)
			print(y_pred_org_1)
			y_pred_org_1 = np.divide(np.subtract(np.subtract(y_pred_org_1, negative_1), negative_0), 10000000)
			y_pred_org_1 = y_pred_org_1.flatten()[-1]
			
			del transformer
			K.clear_session()
			tf.random.set_seed(0)
			tf.compat.v1.reset_default_graph()
			tf.keras.utils.set_random_seed(0)
			tf.config.experimental.enable_op_determinism()
			gc.collect()

			if hist_loss_first == 0:
				hist_loss_first = 1

			learn_perc = ((hist_loss_last-hist_loss_first)/hist_loss_first)*-1
			
			print('Predictions')
			print(datetime.datetime.fromtimestamp(current_time))
			print(current_price)
			print(y_pred_org_1)
			print(learn_perc)


			



		
		except Exception as e:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print(e)
	return y_pred_org_1


if __name__ == '__main__':
	function_1()
