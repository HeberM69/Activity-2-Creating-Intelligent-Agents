import agentpy as ap
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Parte A: Definición de las clases EnhancedPerson y EnhancedVirusModel con las funciones requeridas

class EnhancedPerson(ap.Agent):
    
    def setup(self):
        self.condition = 0  
        self.internal_states = {'moving', 'interacting', 'resting'}  
        self.internal_state = 'resting' 

    def see(self):
        infected_neighbors = sum(1 for neighbor in self.network.neighbors(self) if neighbor.condition == 1)
        if infected_neighbors > 0:
            self.internal_state = 'interacting'
        else:
            self.internal_state = 'moving'

    def next(self):
        if self.internal_state == 'interacting':
            self.decision = 'avoid'
        elif self.internal_state == 'moving':
            self.decision = 'move'
        else:
            self.decision = 'rest'

    def action(self):
        if self.decision == 'avoid' and self.condition == 0:
            pass
        elif self.decision == 'move':
            pass
        elif self.decision == 'rest':
            pass

    def being_sick(self):
        rng = self.model.random
        if self.condition == 1:
            for neighbor in self.network.neighbors(self):
                if neighbor.condition == 0 and self.p.infection_chance > rng.random():
                    neighbor.condition = 1
            if self.p.recovery_chance > rng.random():
                self.condition = 2

class EnhancedVirusModel(ap.Model):

    def setup(self):
        graph = nx.watts_strogatz_graph(
            self.p.population,
            self.p.number_of_neighbors,
            self.p.network_randomness)
        
        self.agents = ap.AgentList(self, self.p.population, EnhancedPerson)
        self.network = self.agents.network = ap.Network(self, graph)
        self.network.add_agents(self.agents, self.network.nodes)
        
        I0 = int(self.p.initial_infection_share * self.p.population)
        self.agents.random(I0).condition = 1 

    def update(self):
        for i, c in enumerate(('S', 'I', 'R')):
            n_agents = len(self.agents.select(self.agents.condition == i))
            self[c] = n_agents / self.p.population
            self.record(c)
        
        if self.I == 0:
            self.stop()

    def step(self):
        self.agents.select(self.agents.condition == 1).being_sick()
        for agent in self.agents:
            agent.see()
            agent.next()
            agent.action()

    def end(self):
        self.report('Total share infected', self.I + self.R)
        self.report('Peak share infected', max(self.log['I']))

# Parte B: Definición de la función de utilidad y la visualización de los resultados

def calculate_utility(model):
    return len(model.agents.select(model.agents.condition != 1))

parameters = {
    'population': 100,
    'infection_chance': 0.3,
    'recovery_chance': 0.1,
    'initial_infection_share': 0.1,
    'number_of_neighbors': 2,
    'network_randomness': 0.5
}

model = EnhancedVirusModel(parameters)
results = model.run()

utility = calculate_utility(model)

def virus_stackplot(data):
    x = data.index.get_level_values('t')
    y = [data[var] for var in ['S', 'I', 'R']]
    plt.stackplot(x, y, labels=['Infected', 'Susceptible', 'Recovered'], colors=['r', 'b', 'g'])
    plt.legend()
    plt.xlabel("Time steps")
    plt.ylabel("Percentage of population")

virus_stackplot(results.variables.EnhancedVirusModel)
plt.show()


