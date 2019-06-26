import tensorflow as tf

a = tf.Variable(tf.random_normal([5, 5]))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    b = a.eval()

    row, col = b.shape
    f = open('embedding', 'w')
    for i in range(row):
        row_list = []
        for j in range(col):
            row_list.append(b[i][j])
        f.write(' '.join(str(x) for x in row_list) + '\n')
    f.close()
