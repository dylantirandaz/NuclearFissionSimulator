import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from mpl_toolkits.mplot3d import Axes3D

# Constants
NEUTRONS_PER_FISSION = {
    'U235': 2.43,
    'U238': 2.3,
    'Pu239': 2.88
}
MEAN_FREE_PATH = {
    'U235': 2.0,
    'U238': 2.5,
    'Pu239': 1.8
}  # cm
FISSION_PROBABILITY = {
    'U235': 0.85,
    'U238': 0.15,
    'Pu239': 0.87
}
CRITICAL_MASS_THRESHOLD = 1000
MAX_NEUTRONS = 100000
TEMPERATURE = 300  # Kelvin
BOLTZMANN_CONSTANT = 8.617333262145e-5  # eV/K

class Isotope:
    def __init__(self, name, density, cross_section):
        self.name = name
        self.density = density
        self.cross_section = cross_section

class Neutron:
    def __init__(self, position, direction, energy):
        self.position = position
        self.direction = direction
        self.energy = energy
        self.age = 0  # Track neutron age

    def move(self, distance):
        self.position += distance * self.direction
        self.age += 1

    def scatter(self, temperature):
        # Implement more realistic scattering using Maxwell-Boltzmann distribution
        maxwell_dist = maxwell(scale=np.sqrt(BOLTZMANN_CONSTANT * temperature / 2))
        speed = maxwell_dist.rvs()
        
        theta = np.arccos(1 - 2 * np.random.random())
        phi = 2 * np.pi * np.random.random()
        
        self.direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        self.energy = 0.5 * speed**2  # E = 1/2 * m * v^2, assuming m=1 for simplicity

class NuclearFissionModel:
    def __init__(self, isotopes, geometry):
        self.neutrons = []
        self.isotopes = isotopes
        self.geometry = geometry
        self.time_steps = []
        self.reaction_rates = []
        self.temperature = TEMPERATURE
        self.fission_events = []

    def initialize_neutrons(self, num_neutrons):
        for _ in range(num_neutrons):
            position = self.geometry.random_point()
            direction = np.random.uniform(-1, 1, 3)
            direction /= np.linalg.norm(direction)
            energy = np.random.uniform(0.025, 2.0)  # eV
            self.neutrons.append(Neutron(position, direction, energy))

    def simulate_transport(self):
        new_neutrons = []
        for neutron in self.neutrons:
            isotope = self.select_isotope()
            distance = np.random.exponential(MEAN_FREE_PATH[isotope.name])
            neutron.move(distance)
            
            if not self.geometry.is_inside(neutron.position):
                continue  # Neutron escaped
            
            if np.random.rand() < FISSION_PROBABILITY[isotope.name]:
                self.fission_events.append(neutron.position)
                for _ in range(int(np.random.normal(NEUTRONS_PER_FISSION[isotope.name], 0.5))):
                    direction = np.random.uniform(-1, 1, 3)
                    direction /= np.linalg.norm(direction)
                    energy = self.fission_energy_spectrum()
                    new_neutrons.append(Neutron(neutron.position, direction, energy))
            else:
                neutron.scatter(self.temperature)
                
        self.neutrons.extend(new_neutrons)
        if len(self.neutrons) > MAX_NEUTRONS:
            self.neutrons = self.neutrons[:MAX_NEUTRONS]

    def select_isotope(self):
        total_density = sum(isotope.density for isotope in self.isotopes)
        r = np.random.random() * total_density
        for isotope in self.isotopes:
            r -= isotope.density
            if r <= 0:
                return isotope
        return self.isotopes[-1]

    def fission_energy_spectrum(self):
        # Implement Watt fission spectrum
        a, b = 0.988, 2.249  # Parameters for U-235
        x = np.random.random()
        return a * np.sinh(np.sqrt(b * x))

    def reaction_rate(self):
        return len(self.neutrons) * sum(isotope.density * isotope.cross_section for isotope in self.isotopes)

    def update_temperature(self):
        # Simple temperature model based on reaction rate
        self.temperature += 0.01 * self.reaction_rate()

    def criticality_analysis(self, steps):
        for step in range(steps):
            self.simulate_transport()
            self.update_temperature()
            rate = self.reaction_rate()
            self.reaction_rates.append(rate)
            self.time_steps.append(step)
            print(f"Step {step}: {rate:.2f} reactions, {len(self.neutrons)} neutrons, T={self.temperature:.2f}K")
            if rate > CRITICAL_MASS_THRESHOLD:
                print(f"System became critical at step {step}")
                break
        return self.reaction_rates

class Geometry:
    def random_point(self):
        raise NotImplementedError

    def is_inside(self, point):
        raise NotImplementedError

class SphereGeometry(Geometry):
    def __init__(self, radius):
        self.radius = radius

    def random_point(self):
        r = self.radius * np.cbrt(np.random.random())
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        return r * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

    def is_inside(self, point):
        return np.linalg.norm(point) <= self.radius

def monte_carlo_simulation(num_neutrons, isotopes, geometry, steps=100):
    model = NuclearFissionModel(isotopes, geometry)
    model.initialize_neutrons(num_neutrons)
    reaction_rates = model.criticality_analysis(steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(model.time_steps, reaction_rates, label='Reaction Rate')
    ax1.axhline(y=CRITICAL_MASS_THRESHOLD, color='r', linestyle='--', label='Critical Mass Threshold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Reaction Rate')
    ax1.legend()
    
    # 3D plot of fission events
    ax2 = fig.add_subplot(122, projection='3d')
    fission_events = np.array(model.fission_events)
    if len(fission_events) > 0:
        ax2.scatter(fission_events[:, 0], fission_events[:, 1], fission_events[:, 2], c='r', marker='o', s=10)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Fission Event Locations')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    isotopes = [
        Isotope('U235', density=0.5, cross_section=585),
        Isotope('U238', density=0.3, cross_section=2.7),
        Isotope('Pu239', density=0.2, cross_section=748)
    ]
    geometry = SphereGeometry(radius=10)  # 10 cm radius sphere
    monte_carlo_simulation(num_neutrons=100, isotopes=isotopes, geometry=geometry, steps=200)
