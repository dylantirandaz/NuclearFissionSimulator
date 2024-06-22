import numpy as np
import matplotlib.pyplot as plt

# Constants
NEUTRONS_PER_FISSION = 2.5
MEAN_FREE_PATH = 1.0  # Arbitrary units
FISSION_PROBABILITY_U235 = 0.05
FISSION_PROBABILITY_U238 = 0.01
CRITICAL_MASS_THRESHOLD = 100
MAX_NEUTRONS = 10000

class Neutron:
    def __init__(self, position, direction, energy):
        self.position = position
        self.direction = direction
        self.energy = energy

    def move(self, distance):
        self.position += distance * self.direction

    def scatter(self):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        self.direction = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        self.energy *= np.random.uniform(0.7, 0.9)  # Lose some energy on scatter

class NuclearFissionModel:
    def __init__(self, material_density_U235, material_density_U238):
        self.neutrons = []
        self.material_density_U235 = material_density_U235
        self.material_density_U238 = material_density_U238
        self.time_steps = []
        self.reaction_rates = []

    def initialize_neutrons(self, num_neutrons):
        for _ in range(num_neutrons):
            position = np.array([0.0, 0.0, 0.0])
            direction = np.random.uniform(-1, 1, 3)
            direction /= np.linalg.norm(direction)
            energy = np.random.uniform(0.5, 2.0)  # Arbitrary energy units
            self.neutrons.append(Neutron(position, direction, energy))

    def simulate_transport(self):
        new_neutrons = []
        for neutron in self.neutrons:
            distance = np.random.exponential(MEAN_FREE_PATH)
            neutron.move(distance)
            if np.random.rand() < FISSION_PROBABILITY_U235:
                for _ in range(int(NEUTRONS_PER_FISSION)):
                    direction = np.random.uniform(-1, 1, 3)
                    direction /= np.linalg.norm(direction)
                    energy = np.random.uniform(0.5, 2.0)
                    new_neutrons.append(Neutron(neutron.position, direction, energy))
            elif np.random.rand() < FISSION_PROBABILITY_U238:
                for _ in range(int(NEUTRONS_PER_FISSION)):
                    direction = np.random.uniform(-1, 1, 3)
                    direction /= np.linalg.norm(direction)
                    energy = np.random.uniform(0.5, 2.0)
                    new_neutrons.append(Neutron(neutron.position, direction, energy))
            neutron.scatter()
        self.neutrons.extend(new_neutrons)
        if len(self.neutrons) > MAX_NEUTRONS:
            self.neutrons = self.neutrons[:MAX_NEUTRONS]

    def reaction_rate(self):
        return len(self.neutrons) * (self.material_density_U235 + self.material_density_U238)

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

def monte_carlo_simulation(num_neutrons, material_density_U235, material_density_U238, steps=100):
    model = NuclearFissionModel(material_density_U235, material_density_U238)
    model.initialize_neutrons(num_neutrons)
    reaction_rates = model.criticality_analysis(steps)

    plt.plot(model.time_steps, reaction_rates, label='Reaction Rate')
    plt.axhline(y=CRITICAL_MASS_THRESHOLD, color='r', linestyle='--', label='Critical Mass Threshold')
    plt.xlabel('Time Step')
    plt.ylabel('Reaction Rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    monte_carlo_simulation(num_neutrons=10, material_density_U235=0.5, material_density_U238=0.2, steps=100)
