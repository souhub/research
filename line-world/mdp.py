from enum import Enum
import numpy as np

class State:
    def __init__(self,position:int) -> None:
        self.position=position

    def __repr__(self) -> str:
        return '<State: {}>'.format(self.position)

    def __eq__(self, other) -> bool:
        return self.position==other.position

    def __hash__(self) -> int:
        return hash((self.position))

    def clone(self):
        return State(self.position)

class Action(Enum):
    STEP=1
    STOP=0

class Env:
    def __init__(self,grid,transit_probs) -> None:
        self.grid=grid
        self.transit_probs=transit_probs

    def T(self,s:State,a:Action):
        return self.transit_probs[s][a]

    def R(self,s:State):
        return 1 if self.grid[s.position]==1 else 0

    def terminate(self,s:State):
        return True if self.grid[s.position]==1 else False

    def step(self,s:State,a:Action):
        terminate_flag=self.terminate(s)
        r=self.R(s)
        probs:dict=self.T(s,a)
        next_state_candidates=probs.keys()
        next_s=np.random.choice(list(next_state_candidates),p=list(probs.values()))
        return r,next_s,terminate_flag

class Agent:
    def __init__(self,policy_probs:dict) -> None:
        self.policy_probs=policy_probs

    def policy(self,s:State=None)->dict:
        return self.policy[s]

    def step(self,s:State):

        a=np.random.choice(list(Action),p=list(self.policy_probs.values()))
        return a

grid=[0,0,0,1]
transit_probs={
    State(0):{
        Action.STEP:{
            State(0):0.1,
            State(1):0.7,
            State(2):0.1,
            State(3):0.1
        },
        Action.STOP:{
            State(0):0.7,
            State(1):0.1,
            State(2):0.1,
            State(3):0.1
        }
    },
    State(1):{
        Action.STEP:{
            State(0):0.1,
            State(1):0.1,
            State(2):0.7,
            State(3):0.1
        },
        Action.STOP:{
            State(0):0.1,
            State(1):0.7,
            State(2):0.1,
            State(3):0.1
        }
    },
    State(2):{
        Action.STEP:{
            State(0):0.1,
            State(1):0.1,
            State(2):0.1,
            State(3):0.7
        },
        Action.STOP:{
            State(0):0.1,
            State(1):0.1,
            State(2):0.7,
            State(3):0.1
        }
    },
    # 3についたときに終了するのでどうせ使わないが設定していないとT()でエラーがでるため一応設定
    State(3):{

        Action.STEP:{
            State(0):0.1,
            State(1):0.1,
            State(2):0.7,
            State(3):0.1
        },
        Action.STOP:{
            State(0):0.1,
            State(1):0.1,
            State(2):0.1,
            State(3):0.7
        }
    }
}

policy_probs={
        Action.STEP:0.5,
        Action.STOP:0.5
    }


if __name__=='__main__':

    env=Env(grid,transit_probs)
    agent=Agent(policy_probs)
    s=State(0)

    terminal_flag=False

    i=0
    while not terminal_flag:
        a=agent.step(s)
        r,next_s,terminal_flag=env.step(s,a)
        s=next_s
        i+=1
        print('{}回目：a={}, s={}, r={}'.format(i,a,s,r))


