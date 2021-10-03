from env import Env, State


if __name__ == '__main__':
    grid = [0, 0, 0, 1]
    env = Env(grid)
    V_map = {}

    for i in range(0, 4):
        V_map[State(i)] = env.V(State(i))

    print(V_map)
