import tensorflow as tf
import gym
import numpy as np

num_inputs = 4
num_hidden = 4
num_outputs = 1 # prob to go left,    1- left = right

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32,shape=[None,num_inputs])

hidden_layer_one = tf.layers.dense(X,num_hidden,activation = tf.nn.relu,kernel_initializer=initializer)

hidden_layer_two = tf.layers.dense(hidden_layer_one,num_hidden,activation = tf.nn.relu,kernel_initializer=initializer)

output_layer = tf.layers.dense(hidden_layer_two,num_outputs,activation=tf.nn.sigmoid,kernel_initializer=initializer)

probabilities = tf.concat(axis=1,values=[output_layer,1-output_layer])
action = tf.multinomial(probabilities,num_samples=1)   # no need to use numpy here

init = tf.global_variables_initializer()

epi = 50  # run 50 training epoch
step_limit = 500  # 500 steps in each game
avg_steps = []

env = gym.make('CartPole-v0')

with tf.Session() as sess:

    sess.run(init)

    for i_episode in range(epi):
        obs = env.reset()

        for step in range(step_limit):
            action_val = action.eval(feed_dict={X:obs.reshape(1,num_inputs)})
            #actional_val[0][0]  #return 0 or 1
            observation, reward,done,info = env.step(action_val[0][0])

            if done:
                avg_steps.append(step)
                print("DONE AFTER {} STEPS".format(step))
                break
print("AFTER {} EPISODES, AVERAGE STEPS PER GAME WAS {} ".format(epi,np.mean(avg_steps)))
env.close()
