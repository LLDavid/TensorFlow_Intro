from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
sess=tf.Session()

##feed value
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

#print(sess.run(z, feed_dict={x: 3, y: 4.5}))
#print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

## Iterate dataset
r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
# while True:
#   try:
#     print(sess.run(next_row))
#   except tf.errors.OutOfRangeError:
#     break

## Createing layers
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=2)
y = linear_model(x)
init=tf.global_variables_initializer()
sess.run(init)
#print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6],[2,3,2]]}))

## Feature Columns
# features = {
#     'sales' : [[5], [10], [8], [9]],
#     'department': ['sports', 'sports', 'gardening', 'gardening']}
#
# department_column = tf.feature_column.categorical_column_with_vocabulary_list(
#         'department', ['sports', 'gardening'])
# department_column = tf.feature_column.indicator_column(department_column)
#
# columns = [
#     tf.feature_column.numeric_column('sales'),
#     department_column
# ]
#
# inputs = tf.feature_column.input_layer(features, columns)
#
# var_init = tf.global_variables_initializer()
# table_init = tf.tables_initializer()
# sess = tf.Session()
# sess.run((var_init, table_init))
#
# print(sess.run(inputs))

## training
# Inputs
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
z_true=tf.constant([0],[1],[0],[1],dtype=tf.float32)# try class labels
# Define the model
linear_model=tf.layers.Dense(units=1)
y_pred=linear_model(x)
z_pred=tf.constant([1],[0],[0],[1],dtype=tf.float32)
# Evaluation
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y_pred))
# Manually calculate absolute difference loss
print(sum(abs(sess.run(y_pred)-sess.run(y_true)))/4)
# Built-in Loss
loss_MSE = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
loss_AD=tf.losses.absolute_difference(labels=y_true, predictions=y_pred)
print(sess.run(loss_MSE))
print(sess.run(loss_AD))
# Optimizer
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss_MSE)
for i in range(100):
    _, loss_value=sess.run((train, loss_MSE))
    print(loss_value)
print(sess.run(y_pred),sess.run(y_true))