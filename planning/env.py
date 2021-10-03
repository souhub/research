from enum import Enum


class Action(Enum):
    STEP = 1
    STOP = 0


class State:
    def __init__(self, position: int, cnt: int = 0) -> None:
        self.position = position
        self.cnt = cnt

    def __repr__(self) -> str:
        return '<State: {}>'.format(self.position)

    def __eq__(self, other) -> bool:
        return self.position == other.position

    def __hash__(self) -> int:
        return hash((self.position))

    def clone(self):
        return State(self.position, self.cnt)


class Env:
    def __init__(self, grid) -> None:
        self.grid = grid

    def V(self, s: State):
        gamma = 0.99
        return self.R(s)+gamma*self.max_next_V(s)

    def terminate(self, s: State):
        return True if self.grid[s.position] == 1 or s.cnt >= 5 else False

    def R(self, s: State):
        return 1 if self.grid[s.position] == 1 else 0

    def T(self, s: State, a: Action):
        next_s = s.clone()
        s = s.clone()
        next_s.position += 1
        if a == Action.STEP:
            return {
                next_s: 0.9,
                s: 0.1
            }
        elif a == Action.STOP:
            return {
                next_s: 0.1,
                s: 0.9
            }
        else:
            raise Exception('Error: Transition function')

    def max_next_V(self, s: State):
        # ゴールに着くまたは行動回数上限で終了
        if self.terminate(s):
            return 0
        s.cnt += 1

        next_V_candidates = []
        for a in Action:
            probs = self.T(s, a)
            q = 0
            for next_s in probs:
                prob = probs[next_s]
                q += prob*self.V(next_s)

            next_V_candidates.append(q)

        return max(next_V_candidates)
