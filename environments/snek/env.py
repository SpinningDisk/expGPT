import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import tensorflow as tf
import math
from keras import layers, Model
import pygame
import functools
import sys
sys.path.insert(0, "/mnt/data/dev/ML/environments/snek/")
from snek import Snake, Apple, drawGrid

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 120}
    
    def __init__(self, render_mode="human", board_size: list|tuple=(16, 16), initial_length: int=4, token_size: int|list|tuple=128):
        self.game_id = 0
        self.agent = Snake(board_size=board_size, initial_length=initial_length, start_pos=[6, 4], dir=[1, 0], token_size=token_size)
        self.board_size = board_size
        self.initial_length = initial_length
        self.token_size = token_size

        self.backpain = 0
        self.past_dir = [1, 0]
        self.screen = None

        self.action_space = spaces.Discrete(4)
        self.clock = None
        self.render_mode = render_mode
    
    def _make_frame(self):
        frame = self.agent.state()
 #       frame = np.expand_dims(frame, -2)
        return frame
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = Snake(self.board_size, self.initial_length, start_pos=[6, 4], dir=[1, 0], token_size=self.token_size)
        # self.agent = Snake(self.board_size, self.initial_length, start_pos=[6, 4], dir=[1, 0], self.token_size)
        frame = self._make_frame()
        self.prev_frame = frame.copy()
        self.backpain = 0
        self.past_dir = [1, 0]

        return np.stack([self.prev_frame, frame]), {}

    def step(self, action):
        reward_mul = 0.1
        reward = 0
        past_score = self.agent.score
        self.agent.eat()
        
        reward += (self.agent.score - past_score) * 10 * self.agent.score**2

        death = self.agent.check_collision()
        done = False
        if death==1:
            print(f"{'\033[1;34m'}AGENT DEATH\ncause: {'\033[0;32m'}hit wall{'\033[0m'}\n")
            reward -= 10
            done = True
            new_frame = self.prev_frame
        elif death==2:
            print(f"{'\033[1;34m'}AGENT DEATH\ncause: {'\033[0;32m'}hit self{'\033[0m'}\n")
            reward -= 200/(self.agent.score+1e-2)
            done = True
            new_frame = self.prev_frame
        else:
            self.agent.dir = action
            self.agent.move()
            new_frame = self._make_frame()

        distances = []
        for apple in self.agent.goal.pos:
            distances.append(math.sqrt((self.agent.pos[0][0]+apple[0])**2 + (self.agent.pos[0][0]+apple[0])**2))
        lowest_distance = min(distances) 
        prev_dist = np.sum(np.abs(np.array(self.agent.pos[1]) - np.array(self.agent.goal.pos[distances.index(lowest_distance)])))
        if prev_dist - lowest_distance < 0:
            reward -= abs(prev_dist - lowest_distance)
        else:
            reward += (prev_dist - lowest_distance) * 2
        

        obs = np.stack([self.prev_frame, new_frame])

        equals = np.equal(self.prev_frame, new_frame)
        not_equal = 0
        for i in equals[..., 0]:
            for j in i:
                not_equal += not j
        if not not_equal:
#            print(f"{'\033[1;35m'}agent aged{'\033[0m'}")
            self.backpain += 1
            reward -= 2
        else:
            backpain = 0
        if self.backpain>=30:
            print(f"{'\033[1;34m'}AGENT DEATH\ncause: {'\033[1;32m'}agent died of old age at {self.backpain}y{'\033[0m'}")
            reward -= 100
            done = True
        self.prev_frame = new_frame.copy()
        return obs, reward*reward_mul, done, False, {}
    def render(self):
        if self.render_mode != "human":
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((2*self.board_size[0]**2, 2*self.board_size[0]**2))

        self.screen.fill((0, 200, 30))
        self.agent.draw(screen=self.screen, blocksize=self.board_size[0])
        drawGrid(self.board_size[0], 3, self.screen)
        pygame.display.update()
    def close(self):
        if self.screen:
            pygame.quit()


if __name__ == "__main__":
    import sys
    import random
    sys.path.insert(0, "/mnt/data/dev/ML/lib")
    from SelectiveTransformation import SelectiveTransformation

    import keras
    import tensorflow as tf
    import math
    import numpy as np

    import pandas as pd
    env = SnakeEnv(token_size=128)

    inputs = tf.keras.layers.Input((2, 18, 18, 3))
    l1 = tf.keras.layers.Lambda(lambda x: tf.concat(x, -1), output_shape=(18, 18, 6))(inputs)
    # l1 = tf.keras.layers.Concatenate(0)([inputs[0][0], inputs[0][1]]),
    l2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3))(l1)
    l3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(l2)
    l4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(l3)
    l5 = tf.keras.layers.Flatten()(l4)
    l6 = tf.keras.layers.Dense(units=64, activation="relu")(l5)
    l7 = tf.keras.layers.Dense(units=4, activation="softmax")(l6)
    model = tf.keras.Model(inputs=inputs, outputs=l7)

    def rbaseline(token_size, token_num):
        input_layer = layers.Input(shape=(2, 18, 18, 1, token_size))  # (frames, H, W, C)
        
        transformed = SelectiveTransformation(keys=[[x for _ in range(token_size)] for x in range(token_num)], input_size=(2, 18, 18, 1, token_size), key_shape=(1, env.token_size))(input_layer)
        transformed = layers.Reshape((2, 18, 18, token_size))(input_layer)

        x = layers.Permute((2, 3, 1, 4))(transformed)  # (18, 18, 2, 3)
        x = layers.Reshape((18, 18, 2*token_size))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.Flatten()(x)

        x = layers.Dense(128, activation='relu')(x)
        output = layers.Dense(4, activation='softmax')(x)  # [up, down, left, right]

        model = Model(inputs=input_layer, outputs=output)
        return model
    model = rbaseline(env.token_size, 5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    gamma = 0.95
    epsilon = 0.05 
    epochs = 5000

    rewards = []
    total_rewards = []
    losses = []
    total_losses = []
    ages = []

    for episode in range(epochs):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        age = 0
        while not done:
            age += 1
            env.render()
            obs_input = np.expand_dims(obs, axis=(0, 4)) / 255.0
            logits = model(obs_input, training=True)
            probs = logits[0].numpy()
            probs /= probs.sum()
            if random.uniform(0, 1) < epsilon:
                direction = [random.randint(-1, 1), random.randint(-1, 1)]
            else:
                action = np.random.choice(len(probs), p=probs)
                match action:
                    case 0:
                        direction = [1, 0]
                    case 1:
                        direction = [-1, 0]
                    case 2:
                        direction = [0, 1]
                    case 3:
                        direction = [0, -1]
            next_obs, reward, done, _, _ = env.step(direction)
            next_obs_input = np.expand_dims(next_obs, axis=(0, 4)) / 255.0  
            rewards.append(reward)

            with tf.GradientTape() as tape:
                q_values = model(obs_input, training=True)
                target_q = q_values.numpy()
                target_q[0, action] = reward + (0 if done else gamma * np.max(model(next_obs_input).numpy()))
                loss = tf.reduce_mean(tf.square(q_values - target_q))
                total_loss += loss
                losses.append(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            obs = next_obs
            total_reward += reward
        total_losses.append(total_loss)
        total_rewards.append(total_reward)
        ages.append(age)
        print(f"{'\033[1;36m'}Episode {episode} - Total Reward: {'\033[1;30m'}{total_reward:.2f}{'\033[1;36m'} - Loss: {'\033[1;31m'}{loss.numpy():.4f}{'\033[0m'}")


    print("saving metrics")
    import matplotlib.pyplot as plt
    window = 50  # or 100, depends on how much smoothness you like
    rewards_smooth = pd.Series(total_rewards).rolling(window).mean()
    losses_smooth = pd.Series(total_losses).rolling(window).mean()

    plt.plot([x for x in range(epochs)], rewards_smooth, color="red", label = "loss")
    plt.plot([x for x in range(epochs)], losses_smooth, color="green", label = "reward")
    plt.plot([x for x in range(epochs)], ages, color="yellow", label = "age")
    plt.savefig("without_st.png")
