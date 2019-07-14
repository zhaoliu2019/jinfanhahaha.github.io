import tensorflow as tf
# 定义常量 hello
hello = tf.constant("Hello TensorFlow")
# 创建一个会话
sess = tf.Session()
# 执行常量hello并打印到标准输出
sess.run(hello)
