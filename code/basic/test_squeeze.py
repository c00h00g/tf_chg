import tensorflow as tf

#用法：从张量形状中移除大小为1的维度

a = tf.constant([[1], [2], [3]])
#(3,1)
print(a.shape)

#将1移除后，只剩下3
#(3,)
b = tf.squeeze(a)
print(b.shape)

#[1 2 3]
with tf.Session() as tf:
    print(tf.run(b))
