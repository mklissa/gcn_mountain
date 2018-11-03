import numpy as np
from gym import core, spaces
from gym.envs.registration import register
import pdb

class Fourrooms:
    def __init__(self):
        layout = """\
wwwwwwwwwwwww
wsssssw     w
wsssssw     w
wsssss      w
wsssssw     w
wsssssw     w
ww wwww     w
w     www www
w     w     w
w     w     w
w        g  w
w     w     w
wwwwwwwwwwwww
"""


        layout = """\
wwwwwwwwwwwwwwwwwwwwww
wsssssssssw          w
wsssssssssw          w
wsssssssssw          w
wsssssssssw          w
wsssssssss           w
wsssssssss           w
wsssssssssw          w
wsssssssssw          w
wsssssssssw          w
wwww  wwwwwwwww  wwwww
w         w          w
w         w          w
w         w          w
w         w          w
w         w          w
w              g     w
w                    w
w         w          w
w         w          w
w         w          w
wwwwwwwwwwwwwwwwwwwwww
"""


#         layout = """\
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssss                      w
# wssssssssssssssssssss                      w
# wssssssssssssssssssss                      w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wwwwwwww     wwwwwwwwwwwwwwwww     wwwwwwwww
# w                    w                     w
# w                    w                     w
# w                    w                     w
# w                    w                     w
# w                                          w
# w                              g           w
# w                                          w
# w                                          w
# w                    w                     w
# w                    w                     w
# w                    w                     w
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# """



#         layout = """\
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssss                      w
# wssssssssssssssssssss                      w
# wssssssssssssssssssss                      w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wsssssssssssssssssssswwwwwww   wwwwww. wwwww
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wwwwwwww     wwwwwwwwwwwwwwwww     wwwwwwwww
# w                    w                     w
# w                    w                     w
# w                    w                     w
# wwww     wwwwwwwwwwwww                     w
# w                                          w
# w                              g           w
# w                                          w
# w                                          w
# w                    w                     w
# w                    w                     w
# w                    w                     w
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# """



#         layout = """\
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# w        ssssssssssssw                     w
# w        ssssssssssssw   wwwwwwwwwwwwww    w
# w    wwwwwwwwwwwwwwwww   w                 w
# w    w               w   w  wwwwwwwwww     w
# w    w    wwwwwwwwwwww   w           w     w
# w    w       w           wwwwwwwwwww w     w
# w    w  w    w   wwwww   w           w     w
# w    w  w    w   w   w   wwwwwwwwwwwww     w
# w    w  w    w   w   w               w     w
# w    w  w    w   w   wwwwwww   wwwwwwwwwwwww
# w       w    w   w   w     w   w           w
# w       w    w   w   w  w  w   w  wwwwww   w
# w    w  w    w   w   w  w  w   w  w        w
# w    w  w    w   w   w  w      w  w wwww   w
# w    w  w            w  w  w      w    w   w
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww wwwww
# w                    w                     w
# wwwwwwwwwwwwwwwww    w                     w
# w                    w                     w
# wwww     wwwwwwwwwwwww                     w
# w        w           w                     w
# w  w     w  w  w  w  w         g           w
# w  w     w  wwwwwwwwww                     w
# w  w     w           w                     w
# w  w     wwwwwwwww   w                     w
# w  w     w           w                     w
# w  w                 w                     w
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# """


        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])


        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)


        # pdb.set_trace()

        self.tostate = {}
        statenum = 0
        for i in range(layout.count('\n')):
            for j in range(len(layout.splitlines()[0])):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum +=  1
        self.tocell = {v:k for k,v in self.tostate.items()}

        # pdb.set_trace()

        goal =np.array([list(map(lambda c: 1 if c=='g' else 0, line)) for line in layout.splitlines()]).flatten()
        

        
        init_states = np.array([list(map(lambda c: 1 if c=='s' else 0, line)) for line in layout.splitlines()]).flatten()
        grid_dict = dict(zip(np.argwhere(self.occupancy.flatten()==0).squeeze(),
                                                range(self.observation_space.n) ))


        self.init_states = map(grid_dict.get,np.argwhere(init_states == 1).flatten())
        self.goal = map(grid_dict.get,np.argwhere(goal == 1).flatten())[0]

        # pdb.set_trace()

        self.state_dict = {v:k for k,v in grid_dict.items()}
        
        # self.init_states = list(range(self.observation_space.n))
        # self.init_states.remove(self.goal)

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        # pdb.set_trace()
        state =self.state_dict.get(state)
        return state

    def step(self, action):
 
        # pdb.set_trace()

        if self.rng.uniform() < 0.:
            empty_cells = self.empty_around(self.currentcell)
            nextcell = empty_cells[self.rng.randint(len(empty_cells))]
        else:
            nextcell = tuple(self.currentcell + self.directions[action])

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = state == self.goal

        state = self.state_dict.get(state)
        return state, float(done), done, None

# env.state_dict.get(env.tostate[env.currentcell])
# register(
#     id='Fourrooms-v0',
#     entry_point='fourrooms:Fourrooms',
#     timestep_limit=20000,
#     reward_threshold=1,
# )
# aa = Fourrooms()

