# Model design
import agentpy as ap
import numpy as np
import random  # Import the random module

# Visualization
import seaborn as sns
from matplotlib import pyplot as plt


class WealthAgent(ap.Agent):
    """ An agent with wealth """
    internal_states = {
        'poor': 0,
        'middle_class': 1,
        'wealthy': 2
    }

    def setup(self):
        self.wealth = np.random.uniform(0, 2)
        # Set internal state based on wealth
        self.state = self.calculate_state()

    def calculate_state(self):
        # Example: Internal state depends on wealth
        if self.wealth < 0.5:
            return self.internal_states['poor']
        elif 0.5 <= self.wealth < 1.5:
            return self.internal_states['middle_class']
        else:
            return self.internal_states['wealthy']

    def wealth_transfer(self):
        if self.wealth > 0:
            partner = self.model.agents.random()
            partner.wealth += 1
            self.wealth -= 1

    def see(self):
        # Agents perceive the average wealth of all agents in the simulation
        all_agents_wealth = [agent.wealth for agent in self.model.agents]
        return np.mean(all_agents_wealth)

    def next(self):
        # Agents decide to transfer wealth based on their internal state
        if self.state == self.internal_states['wealthy']:
            self.action()  # Always transfer if wealthy
        elif self.state == self.internal_states['middle_class']:
            # There is a chance to transfer if middle class
            chance_to_transfer = 0.5  # You can adjust this probability
            if random.random() < chance_to_transfer:
                self.action()  # Transfer wealth
            else:
                pass  # No action if not transferring
        else:
            pass  # No action if poor

    def action(self):
        # Perform wealth transfer
        self.wealth_transfer()
        # Update internal state after wealth transfer
        self.state = self.calculate_state()


def gini(x):
    """ Calculate Gini Coefficient """
    # By Warren Weckesser https://stackoverflow.com/a/39513799

    x = np.array(x)
    mad = np.abs(np.subtract.outer(x, x)).mean()  # Mean absolute difference
    rmad = mad / np.mean(x)  # Relative mean absolute difference
    return 0.5 * rmad


def utility(agents):
    """ Count the number of agents in the 'poor' internal state """
    return sum(agent.state == WealthAgent.internal_states['wealthy'] for agent in agents)


class WealthModel(ap.Model):
    """ A simple model of random wealth transfers """

    def setup(self):
        self.agents = ap.AgentList(self, self.p.agents, WealthAgent)

    def step(self):
         #self.agents.wealth_transfer()
        self.agents.next()

    def update(self):
        #self.record('Gini Coefficient', gini(self.agents.wealth))
        self.record('Poor', utility(self.agents))
        #print(utility(self.agents))

    def end(self):
        self.agents.record('wealth')


if __name__ == '__main__':
    parameters = {
        'agents': 500,
        'steps': 100,
        'seed': 42,
    }

    model = WealthModel(parameters)
    results = model.run()
    print(results.info)
    print(results.variables.WealthModel.head())

    data = results.variables.WealthModel
    ax = data.plot()
    plt.show()

    sns.histplot(data=results.variables.WealthAgent, binwidth=1)

    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
