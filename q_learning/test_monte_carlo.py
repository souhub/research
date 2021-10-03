from monte_carlo import MonteCarloAgent
import pytest


@pytest.fixture
def agent():
    return MonteCarloAgent(gamma=0.99, alpha=0.2)


def test_policy(agent):
    assert agent.policy() in [0, 1, 2]
