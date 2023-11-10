# Model design
import agentpy as ap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython

class Trees(ap.Agent):

    internal_states = {
        'alive': 0,
        'burning': 1,
        'burned': 2
    }

    def setup(self):
        self.state = self.internal_states['alive']

    def see(self):
        neighbors = self.model.forest.neighbors(self)
        return any(neighbor.state == self.internal_states['burning'] for neighbor in neighbors)

    def next(self):
        if self.state == self.internal_states['alive']:
            neighbors = self.model.forest.neighbors(self)
            for neighbor in neighbors:
                if neighbor.state == self.internal_states['burning']:
                    self.state = self.internal_states['burning']
                    return

    def action(self):
        if self.state == self.internal_states['burning']:
            self.state = self.internal_states['burned']

class ForestModel(ap.Model):

    def setup(self):

        # Create agents (trees)
        n_trees = int(self.p['Tree density'] * (self.p.size**2))
        trees = self.agents = ap.AgentList(self, n_trees, Trees)

        # Create grid (forest)
        self.forest = ap.Grid(self, [self.p.size]*2, track_empty=True)
        self.forest.add_agents(trees, random=True, empty=True)

        # Start a fire from the left side of the grid
        unfortunate_trees = self.forest.agents[0:self.p.size, 0:2]
        for tree in unfortunate_trees:
            tree.state = 1

    def step(self):

        # Select burning trees
        burning_trees = self.agents.select(self.agents.state == 1)

        # Spread fire
        for tree in burning_trees:
            for neighbor in self.forest.neighbors(tree):
                if neighbor.state == 0:
                    neighbor.state = 1  # Neighbor starts burning
            tree.state = 2  # Tree burns out

        # Stop simulation if no fire is left
        if len(burning_trees) == 0:
            self.stop()

    def end(self):

        # Document a measure at the end of the simulation
        burned_trees = len(self.agents.select(self.agents.state == 2))
        self.report('Percentage of burned trees',
                    burned_trees / len(self.agents))

# Create single-run animation with custom colors

def animation_plot(model, ax):
    attr_grid = model.forest.attr_grid('state')
    color_dict = {0: '#7FC97F', 1: '#d62c2c', 2: '#e5e5e5', None: '#d5e5d5'}
    ap.gridplot(attr_grid, ax=ax, color_dict=color_dict, convert=True)
    ax.set_title(f"Simulation of a forest fire\n"
                 f"Time-step: {model.t}, Trees left: "
                 f"{len(model.agents.select(model.agents.state == 0))}")

# Define parameters

parameters = {
    'Tree density': 0.6,  # Percentage of grid covered by trees
    'size': 50,  # Height and length of the grid
    'steps': 100,
}

fig, ax = plt.subplots()
model = ForestModel(parameters)
animation = ap.animate(model, fig, ax, animation_plot)
IPython.display.HTML(animation.to_jshtml(fps=15))

# Prepare parameter sample
parameters = {
    'Tree density': ap.Range(0.2, 0.6),
    'size': 100
}
sample = ap.Sample(parameters, n=30)

# Perform experiment
exp = ap.Experiment(ForestModel, sample, iterations=40)
results = exp.run()

# Save and load data
results.save()
results = ap.DataDict.load('ForestModel')

# Plot sensitivity
sns.set_theme()
sns.lineplot(
    data=results.arrange_reporters(),
    x='Tree density',
    y='Percentage of burned trees'
)