import gym

env = gym.make('CartPole-v0')

observation = env.reset()

for t in range(1000):

    env.render()

    cart_pos, cart_vel, pole_ang, ang_vel = observation

    if pole_ang > 0: # pole fall to the right
        action = 1 # cart move to right
    else:          # pole fall to the left
        action = 0 # cart move to left

    observation, reward,done,info = env.step(action)
