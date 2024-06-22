import numpy as np
import matplotlib.pyplot as plt

# Constants
NEUTRONS_PER_FISSION = 2.5
MEAN_FREE_PATH = 1.0  # Arbitrary units
FISSION_PROBABILITY = 0.05  # Probability of fission per collision (reduced)
CRITICAL_MASS_THRESHOLD = 100  # Arbitrary threshold for critical mass (increased)
MAX_NEUTRONS = 10000  # Max neutrons to prevent runaway simulation

class Neutron:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction

    def move(self, distance):
        self.position += distance * self.direction

    def scatter(self):
        theta = np.random.uniform(0, 2 * np.pi)
        self.direction = np.array([np.cos(theta), np.sin(theta)])

class NuclearFissionModel:
    def __init__(self, material_density):
        self.neutrons = []
        self.material_density = material_density
        self.time_steps = []
        self.reaction_rates = []

    def initialize_neutrons(self, num_neutrons):
        for _ in range(num_neutrons):
            position = np.array([0.0, 0.0])
            direction = np.random.uniform(-1, 1, 2)
            direction /= np.linalg.norm(direction)
            self.neutrons.append(Neutron(position, direction))

    def simulate_transport(self):
        new_neutrons = []
        for neutron in self.neutrons:
            distance = np.random.exponential(MEAN_FREE_PATH)
            neutron.move(distance)
            if np.random.rand() < FISSION_PROBABILITY:
                for _ in range(int(NEUTRONS_PER_FISSION)):
                    direction = np.random.uniform(-1, 1, 2)
                    direction /= np.linalg.norm(direction)
                    new_neutrons.append(Neutron(neutron.position, direction))
            neutron.scatter()
        self.neutrons.extend(new_neutrons)
        if len(self.neutrons) > MAX_NEUTRONS:
            self.neutrons = self.neutrons[:MAX_NEUTRONS]  # Prevent runaway simulation

    def reaction_rate(self):
        return len(self.neutrons) * self.material_density

    def criticality_analysis(self, steps):
        for step in range(steps):
            self.simulate_transport()
            rate = self.reaction_rate()
            self.reaction_rates.append(rate)
            self.time_steps.append(step)
            print(f"Step {step}: {rate} reactions, {len(self.neutrons)} neutrons")
            if rate > CRITICAL_MASS_THRESHOLD:
                print(f"System became critical at step {step}")
                break
        return self.reaction_rates

def monte_carlo_simulation(num_neutrons, material_density, steps=100):
    model = NuclearFissionModel(material_density)
    model.initialize_neutrons(num_neutrons)
    reaction_rates = model.criticality_analysis(steps)

    plt.plot(model.time_steps, reaction_rates, label='Reaction Rate')
    plt.axhline(y=CRITICAL_MASS_THRESHOLD, color='r', linestyle='--', label='Critical Mass Threshold')
    plt.xlabel('Time Step')
    plt.ylabel('Reaction Rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    monte_carlo_simulation(num_neutrons=10, material_density=0.5, steps=100)
