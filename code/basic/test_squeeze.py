import tensorflow as tf

#�÷�����������״���Ƴ���СΪ1��ά��

a = tf.constant([[1], [2], [3]])
#(3,1)
print(a.shape)

#��1�Ƴ���ֻʣ��3
#(3,)
b = tf.squeeze(a)
print(b.shape)

#[1 2 3]
with tf.Session() as tf:
    print(tf.run(b))
