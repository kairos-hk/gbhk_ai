import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')


output_layer = model.layers[-1]
output_classes = output_layer.output_shape[-1]

category_index = list(range(output_classes))

print(category_index)