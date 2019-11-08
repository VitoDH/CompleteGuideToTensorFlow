import gym

env = gym.make('CartPole-v0')

print('INITIAL OBSERVATION')
observation = env.reset()
# STATE (pole standing straight up in the cart)
print(observation)

for t in range(2):
    # plot the environment and view it
    #env.render()

    # take some random actions and provide it to the environment
    action = env.action_space.sample()

    observation,reward,done,info = env.step(action)
    print("Performed One Random Action")
    print('\n')
    print('observation')
    print(observation)
    print('\n')

    print('reward')
    print(reward)
    print('\n')

    print('done')
    print(done)
    print('\n')

    print('info')
    print(info)
    print('\n')
