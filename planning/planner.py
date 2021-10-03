from env import Env, State, Action


# 価値反復法と方策反復法のベースとなるクラス
class Planner(Env):

    def transition_at(self, s: State, a: Action):
        probs = self.T(s, a)
        for next_s in probs:
            prob = probs[next_s]
            next_r = self.R(next_s)
            yield prob, next_s, next_r

    def plan(self, gamma=0.9, threshold=0.001):
        raise Exception('Planner have to implement plan method')

    # 端まで到達またはゴール地点のとき終了
    # ゴール地点以降はすべて0となるが、報酬が 0 or 1 であることと、Action が STEP or STOP であることを考えれば必然？
    def terminate(self, s: State):
        return True if self.grid[s.position] == 1 or s.position == len(grid)-1 else False


# 価値反復法
class ValuePlanner(Planner):
    def plan(self, gamma=0.9, threshold=0.00001):
        V = {State(i): 0 for i in range(0, len(grid))}

        while True:
            delta = 0
            for s in V:
                # 終了地点（State(3)）は計算できないというかしないものっぽい？ V(State(3))=1.0 は自明だから？
                # ゴール（報酬の在り処）地点（State(i).position==1）は計算できないというかしないものっぽい？ V(State(goal))=1.0 は自明だから？
                if self.terminate(s):
                    continue

                expected_rewards = []
                for a in Action:
                    q = 0
                    for prob, next_s, next_r in self.transition_at(s, a):
                        q += prob*(next_r+gamma*V[next_s])
                    expected_rewards.append(q)
                max_q = max(expected_rewards)
                delta = max(delta, abs(max_q-V[s]))
                V[s] = max_q

            if delta < threshold:
                break

        print(V)


# 方策反復法
class PolicyPlanner(Planner):
    def __init__(self, grid) -> None:
        self.policy = {}
        self.grid = grid

    def initialize(self):
        self.policy = {State(i): {Action(j): 1/len(Action)
                                  for j in range(0, 2)} for i in range(0, len(grid))}

    def estimate_V_by_policy(self, gamma=0.99, threshold=0.00001):
        V = {State(i): 0 for i in range(0, len(self.grid))}

        while True:
            delta = 0
            for s in V:
                # 終了条件（オリジナル）
                if self.terminate(s):
                    continue

                v = 0
                for a in Action:
                    action_prob = self.policy[s][a]
                    q = 0
                    for prob, next_s, next_r in self.transition_at(s, a):
                        q += prob*(next_r+gamma*V[next_s])
                    v += action_prob*q
                delta = max(delta, abs(v-V[s]))
                V[s] = v
            if delta < threshold:
                break

        return V

    def plan(self, gamma=0.99, threshold=0.00001):
        self.initialize()

        def get_max_action(action_value_dict: dict):
            return max(action_value_dict, key=action_value_dict.get)

        while True:
            update_stable = True
            V = self.estimate_V_by_policy(gamma, threshold)

            for s in V:
                # 終了条件（オリジナル）
                if self.terminate(s):
                    continue

                policy_action = get_max_action(self.policy[s])

                action_rewards = {}
                for a in Action:
                    r = 0
                    for prob, next_s, next_r in self.transition_at(s, a):
                        r += prob*(next_r+gamma*V[next_s])
                    action_rewards[a] = r
                best_action = get_max_action(action_rewards)

                if policy_action != best_action:
                    update_stable = False

                for a in self.policy[s]:
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob

            if update_stable:
                break

        print(V)


def sample_transition_at():
    p = Planner(grid)
    s = State(0)
    a = Action.STEP
    for prob, next_s, next_r in p.transition_at(s, a):
        print(prob, next_s, next_r)


def sample_value_planner():
    grid = [0, 0, 0, 0, 0, 1]
    vp = ValuePlanner(grid)
    vp.plan()


def sample_policy_planner():
    grid = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    pp = PolicyPlanner(grid)
    pp.plan()


grid = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
pp = PolicyPlanner(grid)
pp.initialize()
pp.plan()
