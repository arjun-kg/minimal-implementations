import gym
import pdb
from scipy.misc import imresize
from scipy.misc import imshow
env = gym.make('Pong-v0')

def preprocess(im):
    # image expects 210 x 160 x 3
    im = 0.2989*im[:, :, 0] + 0.5870*im[:, :, 1] + 0.1140*im[:, :, 2]
    im = imresize(im, (110, 84))
    im = im[18:102, :]
    return im

for ep in range(100):
    done = False
    env.reset()
    st = 0
    while not done:
        st+= 1
        obs, _, done, _ = env.step(env.action_space.sample())
        obs = preprocess(obs)
        env.render()
    print(st)
