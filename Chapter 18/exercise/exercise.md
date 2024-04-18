1. Reinforcement learning is a type of machine learning that you do not learn from a predefined dataset, but instead from an environment. There will be a task we want the model to do well, and so wec an define a reward an maximize the reward by changing the model's actions.

Here are some key differences to distinct RL from regular supervised or unsupervised learning:
- In supervised learning and unsupervised learning, you learn the structures in the dataset, which hopefully will generalize well to new data, and use them to make predictions. In RL, we instead want to find a policy that works well in the environment.
- We don't have any dataset, we just have an environment. The agent (i.e., the model) must learn by trial and error.
- Unlike supervised learning, the agent is not given the explicitly "right" answer.
- Unlike unsupervised learning, the agent still has a form of supervision, through rewards. It isn't told the right way to do, but it's told if it's making progress or failing.
- An agent in the RL algorithm must find a balance between exploration and exploitation: it needs to explore the unknown regions of the environment for a potential better reward, and exploit known good regions. In contrast, supervised and unsupervised learning don't need to care about this balance: they just work on the fed dataset.
- In supervised learning and unsupervised learning, the training instances are generally independent, hence we can use gradient descent. In RL, the consecutive observations are typically dependent, as the agent remains on the same region of the environment. In some cases, a replay buffer is used to ensure the agent can have fairly uncorrelated observations.

2. Three possible applications of RL that were not mentioned in the book:
- Music personalization: The environment is a user's personalized radio. The agent is the software deciding which song should be played next. The possible actions are play a song in a catalog (the agent must choose a song the user want to listen) or an advertisement (the agent must choose an ad the user will be interested in). It will gain a huge positive reward if the user watches the advertisement, a positive reward if the user listens to a song, a negative reward if the user skip an ad or a song, and a huge negative reward if the user leaves.
- Marketing: The environment is your company's marketing department. The agent is the software deciding which customer a mailing campaign should be sent to, given the customer's profile and purchase history. For each customer, it can choose either send or not send. It gets a negative reward for the price of the campaign, and a positive reward for the estimated revenue generated from this campaign.
- Product delivery: The environment can be a city. The agent control a set of trucks, deciding what they should pick up at the depots, where they should go, what they should drop off, and so on. It get a positive reward for every product delivered on time, and a negative for every late deliveries.

3.
- The discount factor is a number measured how the future rewards are evaluated relatively compared to the present rewards. For example, if the discount factor is 0.9, then a reward of 100 in two time step ahead is equivalent to a reward of $100 \times 0.9^2 = 81$ in the current time step. Discount factor ranges from 0 to 1, included.
- If you change the discount factor, the optimal policy will be affected tremendously: The higher the discount factor, the more we value the future rewards. In other words, the more we're willing accept the pain in the present for a great gain in the future.

4.
- You can measure the performance of a reinforcement learning agent by summing up all the rewards it have gotten so far. In a stimulated environment, you can rerun the agent many times (without training it), and measure its statistics parameters (mean, median, max, min, standard deviation, and so on).
- Note that if you use Deep Q-learning, then you should not measure the agent's performance by the loss function of the DQN, as it can be very misleading.

5.
- The credit assignment problem is when the agent has no idea which actions are credited for the reward it gets. 
- It occurs when there's a delay between an action and the corresponding reward. For example, in the game of Pong, there's a time delay, hence frame delay between the time the agent hits the ball and the time it gets a point.
- A way to alleviate it is to add intermediate reward (i.e. short-term rewards), if possible. It requires prior knowledge of the task and this intermediate reward must algin with the main reward, or else the agent will act unexpectedly. For example, if you build an agent to play chess, you could give it a reward when it captures any piece, not only when it wins the game, as this would be a huge time delay.

6.
- The agent can often stay in a region for a while, so all of its experiences will be very similar for a period of time. This can introduce some bias to this agent. The agent can work very well in this region, but not so great when leaving this region. In other words, it overfits this region.
- To solve this, you can use a replay buffer. The replay buffer will store lots of it past experiences, recent and not so recent. This can help reduce the corelation between the experiences being fed to the agent, hence reduce the bias. This is similar to the reason why we sleep at night, to process the information being received while awake.

7. An off-policy RL algorithm learns the value of the optimal policy (i.e., the sum of the discounted rewards that cna be expected in each state if the agent plays optimally), while the agent explore by a different policy. Put it more simply, the policy the agent tries to learn and the policy to explore is different. In contrast, in an on-policy RL algorithm, these two policies are the same (they pointed to the memory space), and are learned at the same time.