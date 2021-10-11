import gym
import numpy as np
import matplotlib.pyplot as plt


class Planner:
    """A base class for planners.

    Attributes:
        env: An environment of gym.
        gamma: A learning rate.
        thereshold: A decimal deciding when to finish learning.
    """

    def __init__(self, env: gym.Env, gamma: float = 0.9, threshold: float = 10**(-5)) -> None:
        self.env = env
        self.log = []
        self.gamma = gamma
        self.threshold = threshold
        self.R = None

    def initialize(self):
        """Initialize an environment.
        """
        self.env.reset()

    def plan(self):
        """Implement an algorithm for planners.
        """
        raise Exception("Planner must be implemented")

    def compute_action_values(self, state, V):
        """Compute action values.

        Args:
            state: An integer representing an observation of an environment.
            V: A list of State values.

        Returns:
            A dict mapping actions to acction values.
            examples:

            {0: 0.39556574140843165, 1: 0.6390163154732553, 2: 0.6149209118442326, 3: 0.5371942554993754}
        """
        Q = {}

        for action in range(self.env.action_space.n):
            v = 0
            probs = self.env.P[state][action]
            for prob, next_state, reward, done in probs:
                if self.R is not None:
                    reward = self.R[state]
                v += prob*(reward+self.gamma*V[next_state])
            Q[action] = v
        return Q


class ValueIterationPlanner(Planner):
    """A class implemented value iteration.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def plan(self):
        self.initialize()
        V = {}

        for s in range(self.env.observation_space.n):
            V[s] = 0

        while True:
            delta = 0
            for s in V:
                expected_action_values = self.compute_action_values(
                    s, V)
                max_Q = max(expected_action_values.values())
                delta = max(delta, abs(max_Q-V[s]))
                V[s] = max_Q

            if delta < self.threshold:
                break
        V[15] = 1
        return V


class PolicyIterationPlanner(Planner):
    """A class implemented policy iteration.

    Atributes:
        env: An environment of gym.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.policy = {}

    def initialize(self):
        """Initialize a policy.
        """
        super().initialize()
        self.policy = {}
        for s in range(self.env.nS):
            self.policy[s] = {}
            for a in range(self.env.nA):
                self.policy[s][a] = 1/self.env.nA

    def compute_state_values(self):
        """Compute state values.

        Returns:
            V: A map of state values.
        """
        V = {}
        for s in range(self.env.nS):
            V[s] = 0

        while True:
            delta = 0
            for s in V:
                value = 0
                expected_action_values = self.compute_action_values(
                    s, V)
                action_probs = self.policy[s]
                for (action_prob, action_value) in zip(action_probs.values(), expected_action_values.values()):
                    value += action_prob*action_value
                delta = max(delta, abs(value-V[s]))
                V[s] = value

            if delta < self.threshold:
                break

        V[15] = 1
        return V

    def render_values(self):
        V, policy = self.plan()
        V = np.array(list(V.values()))
        plt.pcolor(V.reshape(4, 4)[::-1, :])
        plt.title("Value function")
        plt.colorbar()
        plt.show()

    def render_policy(self):
        def convert_action(num: int):
            if num == 0:
                return 'L'
            elif num == 1:
                return 'D'
            elif num == 2:
                return 'R'
            else:
                return 'U'
        _, policy = self.plan()
        policy = np.array(list(policy.values()))
        render_policy = []
        for p in policy:
            for action in p:
                if p[action] == 1:
                    render_policy.append(convert_action(action))
        render_policy = np.array(render_policy).reshape(4, 4)[::-1, :]
        print(render_policy)

    def plan(self):
        """An algorithm for plan iteration.
        """
        self.initialize()

        while True:
            update_stable = True

            # Estimate expected values under a current policy
            V = self.compute_state_values()
            for s in range(self.env.observation_space.n):
                policy_action = max(self.policy[s], key=self.policy[s].get)
                expected_action_values = self.compute_action_values(
                    s, V)
                best_action = max(expected_action_values,
                                  key=expected_action_values.get)
                if policy_action != best_action:
                    update_stable = False

                # Update the policy
                for action in range(self.env.nA):
                    prob = 1 if action == best_action else 0
                    self.policy[s][action] = prob

            if update_stable:
                break

        return V, self.policy

    def sample_trajectories(self, n_samples: int, n_steps: int, is_succeeded_data_only: bool = True) -> np.ndarray:
        """Sample trajactories data.

        Sample succeeded expert's data  or both succeeded and failed data.

        Args:
            n_samples: An integer that determines the nuber of trajectories.
            n_steps: An integer that determines the number of time steps.
            is_succeeded_data_only: A boolean that decides whether the data consists of expert's data succeeded only.

        Returns:
            A list of trajectories.
        """
        trajectories = []
        V, expert_policy = self.plan()

        while len(trajectories) < n_samples:
            state = self.env.reset()
            done = False
            trajectory = []
            total_reward = 0
            while not done:
                trajectory.append(state)
                action_probs = expert_policy[state]
                action = np.random.choice(
                    self.env.action_space.n, 1, action_probs.values())[0]
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state

            trajectory.append(state)
            trajectory.extend(
                [state] * (n_steps - len(trajectory)))

            if is_succeeded_data_only:
                condition = (total_reward >= 1 and len(
                    trajectory) == n_steps)
            else:
                condition = (len(trajectory) == n_steps)

            if condition:
                trajectories.append(trajectory)
                if len(trajectories) % 5 == 0:
                    print(
                        '    {}/{} trajectories have benn collected...'.format(len(trajectories), n_samples))

        return np.array(trajectories)
