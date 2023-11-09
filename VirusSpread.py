from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from prettytable import PrettyTable


class Person(Agent):

  def __init__(self, unique_id, model):
    super().__init__(unique_id, model)
    self.infected = False
    self.days_infected = 0
    self.internal_states = {'healthy', 'infected'}

  def step(self):
    # Simulates the transition from healthy to infected
    self.next()
    # Prints the agent's internal state
    self.see()
    # Simulates the action of the infected agent
    self.action()

  def next(self):
    if not self.infected and random.random() < self.model.infection_chance:
      self.infected = True
      self.days_infected = 1

  def see(self):
    table_see = PrettyTable()
    table_see.field_names = ["Agent ID", "Internal States"]
    table_see.add_row([self.unique_id, self.internal_states])
    print(table_see)

  def action(self):
    if self.infected:
      table_action = PrettyTable()
      table_action.field_names = ["Agent ID", "Action"]
      table_action.add_row([self.unique_id, "Taking action as infected"])
      print(table_action)


class VirusSpread(Model):

  def __init__(self, population_size, infection_chance, recovery_chance,
               infection_duration):
    self.population_size = population_size
    self.infection_chance = infection_chance
    self.recovery_chance = recovery_chance
    self.infection_duration = infection_duration
    self.schedule = RandomActivation(self)
    self.grid = MultiGrid(10, 10, True)
    self.datacollector = DataCollector(
        agent_reporters={"Infected": lambda a: a.infected})
    self.steps = 0

    for i in range(self.population_size):
      agent = Person(i, self)
      x = self.random.randrange(self.grid.width)
      y = self.random.randrange(self.grid.height)
      self.grid.place_agent(agent, (x, y))
      self.schedule.add(agent)

  def step(self):
    self.datacollector.collect(self)
    self.schedule.step()
    self.steps += 1


population_size = int(input("Enter the population size: "))
infection_chance = float(input("Enter the infection probability: "))
recovery_chance = float(input("Enter the recovery probability: "))
infection_duration = int(input("Enter the infection duration: "))
num_steps = int(input("Enter the number of time steps: "))

model = VirusSpread(population_size, infection_chance, recovery_chance,
                    infection_duration)

fig, ax = plt.subplots()
ax.set_xlim(0, model.grid.width)
ax.set_ylim(0, model.grid.height)


def init():
  return []


def update(frame):
  model.step()
  ax.clear()
  ax.set_xlim(0, model.grid.width)
  ax.set_ylim(0, model.grid.height)

  infected_agents = [
      agent.pos for agent in model.schedule.agents if agent.infected
  ]
  healthy_agents = [
      agent.pos for agent in model.schedule.agents if not agent.infected
  ]

  ax.scatter(*zip(*infected_agents),
             color='red',
             label='Infected',
             s=50,
             alpha=0.7)
  ax.scatter(*zip(*healthy_agents),
             color='green',
             label='Healthy',
             s=50,
             alpha=0.7)

  ax.legend()
  ax.set_title(
      f"Step: {model.steps} | Infection Duration: {model.infection_duration} | Recovery Probability: {model.recovery_chance}"
  )

  current_percentage = (model.datacollector.get_agent_vars_dataframe().groupby(
      'Step')['Infected'].sum().iloc[-1] / population_size) * 100
  ax.text(0.5,
          -0.1,
          f'Percentage of Infected: {current_percentage:.2f}%',
          transform=ax.transAxes,
          ha='center')


animation = FuncAnimation(fig,
                          update,
                          frames=num_steps,
                          init_func=init,
                          blit=False)

plt.show()

final_percentage = (model.datacollector.get_agent_vars_dataframe().groupby(
    'Step')['Infected'].sum().iloc[-1] / population_size) * 100
print(
    f'The percentage of infected at the end of the simulation is: {final_percentage:.2f}%'
)
