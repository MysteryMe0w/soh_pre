import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

with tf.device("/GPU:0"):
    a = tf.random.normal([8192, 8192])
    b = tf.random.normal([8192, 8192])
    c = tf.matmul(a, b)
print("done on:", c.device)
