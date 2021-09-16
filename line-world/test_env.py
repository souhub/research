from env import State, Env, Action

import pytest


@pytest.fixture
def state():
    return State(4)


def test_clone(state):
    assert state.clone() == State(4)


@pytest.fixture
def env():
    grid = [0, 0, 0, 1]
    return Env(grid)


@pytest.mark.parametrize("state, expected", [
    (State(3), 1),
    (State(2), 0),
    (State(1), 0),
    (State(0), 0)
])
def test_R(env, state, expected):
    assert env.R(state) == expected


@pytest.mark.parametrize("state, expected", [
    (State(3, 6), True),
    (State(3, 0), True),
    (State(0, 5), True),
    (State(0, 0), False),
])
def test_terminate(env, state, expected):
    assert env.terminate(state) == expected


@pytest.mark.parametrize("state, action, expected", [
    (State(0), Action.STEP, {
        State(1): 0.9,
        State(0): 0.1
    }),
    (State(0), Action.STOP, {
        State(1): 0.1,
        State(0): 0.9
    })
])
def test_T(env, state, action, expected):
    assert env.T(state, action) == expected
