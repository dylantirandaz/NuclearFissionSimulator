import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize
from collections import deque


# Constants
NEUTRONS_PER_FISSION = {
    'U235': 2.43, 'U238': 2.3, 'Pu239': 2.88, 'Pu241': 2.93, 'Th232': 2.45
}
MEAN_FREE_PATH = {
    'U235': 2.0, 'U238': 2.5, 'Pu239': 1.8, 'Pu241': 1.9, 'Th232': 2.2
}  # cm
FISSION_PROBABILITY = {
    'U235': 0.085, 'U238': 0.015, 'Pu239': 0.087, 'Pu241': 0.089, 'Th232': 0.01
}
NEUTRON_ABSORPTION_PROBABILITY = 0.2
HEAT_CAPACITY = 0.2  # J/(g·K)
COOLING_RATE = 1e-6  # K/s
CRITICAL_MASS_THRESHOLD = 1000
MAX_NEUTRONS = 1000000
BOLTZMANN_CONSTANT = 8.617333262145e-5  # eV/K

def generate_realistic_cross_section_data():
    isotopes = ['U235', 'U238', 'Pu239', 'Pu241', 'Th232']
    energies = np.logspace(-5, 7, 1000)  # 10^-5 to 10^7 eV
    data = []
    
    for isotope in isotopes:
        if isotope == 'U235':
            base_xs = 100 + 500 * np.exp(-energies / 0.0253)  # Peak at thermal energy (0.0253 eV)
        elif isotope == 'U238':
            base_xs = 10 + 50 * np.exp(-energies / 1e6)
        else:
            base_xs = 50 + 200 * np.exp(-energies / 1e5)
        
        resonances = np.random.uniform(0, 1000, 50)
        resonance_energies = np.random.choice(energies, 50)
        
        for e, re in zip(resonance_energies, resonances):
            base_xs += re * np.exp(-(np.log(energies) - np.log(e))**2 / (2 * 0.1**2))
        
        for energy, xs in zip(energies, base_xs):
            data.append({'isotope': isotope, 'energy': energy, 'cross_section': xs})
    
    return pd.DataFrame(data)

cross_section_data = generate_realistic_cross_section_data()

class Isotope:
    def __init__(self, name, density, abundance):
        self.name = name
        self.density = density
        self.abundance = abundance
        self.cross_section_func = self._load_cross_section()

    def _load_cross_section(self):
        data = cross_section_data[cross_section_data['isotope'] == self.name]
        return interp1d(data['energy'], data['cross_section'], kind='cubic', fill_value='extrapolate')

class Neutron:
    def __init__(self, position, direction, energy):
        self.position = position
        self.direction = direction
        self.energy = energy
        self.age = 0

    def move(self, distance):
        self.position += distance * self.direction
        self.age += 1

    def scatter(self, temperature):
        maxwell_dist = maxwell(scale=np.sqrt(BOLTZMANN_CONSTANT * temperature / 2))
        speed = maxwell_dist.rvs()
        
        mu = 2 * np.random.random() - 1
        phi = 2 * np.pi * np.random.random()
        
        sin_theta = np.sqrt(1 - mu**2)
        self.direction = np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            mu
        ])
        self.energy = 0.5 * speed**2

        self.energy += np.random.normal(0, np.sqrt(2 * BOLTZMANN_CONSTANT * temperature * self.energy))

class Material:
    def __init__(self, isotopes, temperature):
        self.isotopes = isotopes
        self.temperature = temperature
        self.total_density = sum(isotope.density for isotope in isotopes)
        self.heat_capacity = HEAT_CAPACITY * self.total_density

    def select_isotope(self):
        r = np.random.random() * self.total_density
        for isotope in self.isotopes:
            r -= isotope.density
            if r <= 0:
                return isotope
        return self.isotopes[-1]

    def update_temperature(self, energy_deposition, time_step):
        temperature_change = energy_deposition / (self.heat_capacity * self.total_density)
        self.temperature += temperature_change
        self.temperature -= COOLING_RATE * time_step
        self.temperature = max(self.temperature, 300)  # Ensure temperature doesn't go below 300K

class SphericalShellGeometry:
    def __init__(self, inner_radius, outer_radius):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def random_point(self):
        r = np.cbrt(np.random.uniform(self.inner_radius**3, self.outer_radius**3))
        theta = np.arccos(2 * np.random.random() - 1)
        phi = 2 * np.pi * np.random.random()
        return r * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    def is_inside(self, point):
        r = np.linalg.norm(point)
        return self.inner_radius <= r <= self.outer_radius

class NuclearFissionModel:
    def __init__(self, material, geometry):
        self.neutrons = []
        self.material = material
        self.geometry = geometry
        self.time_steps = []
        self.reaction_rates = []
        self.fission_events = []
        self.energy_deposition = 0
        self.time = 0
        self.fission_events = deque(maxlen=10000)  # Store last 10000 events
        self.neutron_lifecycles = {}
        self.energy_deposition = 0


    def initialize_neutrons(self, num_neutrons):
        for _ in range(num_neutrons):
            position = self.geometry.random_point()
            direction = np.random.uniform(-1, 1, 3)
            direction /= np.linalg.norm(direction)
            energy = self.watt_spectrum()
            self.neutrons.append(Neutron(position, direction, energy))

    def reaction_rate(self):
        """Calculate the current reaction rate."""
        return len(self.neutrons) * self.material.total_density

    def random_direction(self):
        """Generate a random unit vector for neutron direction."""
        phi = 2 * np.pi * np.random.random()
        costheta = 2 * np.random.random() - 1
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.array([x, y, z])

    def watt_spectrum(self):
        """Generate neutron energy based on Watt fission spectrum."""
        a, b = 0.988, 2.249  # Parameters for U-235
        x = np.random.random()
        E = -a * np.log(np.random.random()) - b * np.log(np.random.random())
        return a * np.sinh(np.sqrt(b * E))


    def determine_interaction(self, neutron, isotope):
        cross_section = isotope.cross_section_func(neutron.energy)
        fission_prob = FISSION_PROBABILITY[isotope.name] * cross_section / np.max(cross_section_data['cross_section'])
        fission_prob *= 0.001  # Further reduce fission probability
        
        rand = np.random.random()
        if rand < fission_prob:
            return 'fission'
        elif rand < fission_prob + NEUTRON_ABSORPTION_PROBABILITY:
            return 'absorption'
        else:
            return 'scatter'

    def simulate_transport(self, time_step):
        new_neutrons = []
        fissions_this_step = 0
        for neutron in self.neutrons:
            isotope = self.material.select_isotope()
            distance = np.random.exponential(MEAN_FREE_PATH[isotope.name])
            neutron.move(distance * time_step)
            
            if not self.geometry.is_inside(neutron.position):
                continue  # Neutron escaped

            interaction_type = self.determine_interaction(neutron, isotope)
            if interaction_type == 'fission':
                fissions_this_step += 1
                self.fission_events.append((self.time, neutron.position, self.material.temperature))
                self.energy_deposition += 200e6 * 1.60218e-19  # Convert eV to Joules
                for _ in range(int(np.random.normal(NEUTRONS_PER_FISSION[isotope.name], 0.5))):
                    new_neutron = Neutron(neutron.position, self.random_direction(), self.watt_spectrum())
                    new_neutrons.append(new_neutron)
                    self.neutron_lifecycles[id(new_neutron)] = self.time
            elif interaction_type == 'scatter':
                neutron.scatter(self.material.temperature)
                new_neutrons.append(neutron)
            elif interaction_type == 'absorption':
                if id(neutron) in self.neutron_lifecycles:
                    del self.neutron_lifecycles[id(neutron)]

        self.neutrons = new_neutrons
        if len(self.neutrons) > MAX_NEUTRONS:
            self.neutrons = self.neutrons[:MAX_NEUTRONS]

        self.material.update_temperature(self.energy_deposition, time_step)
        self.energy_deposition = 0  # Reset energy deposition for next step
        
        return fissions_this_step
        

    def criticality_analysis(self, total_time, initial_time_step):
        time_step = initial_time_step
        while self.time < total_time:
            fissions = self.simulate_transport(time_step)
            rate = self.reaction_rate()
            self.reaction_rates.append(rate)
            self.time_steps.append(self.time)
            
            print(f"Time: {self.time:.4f}s, Fissions: {fissions}, Neutrons: {len(self.neutrons)}, T={self.material.temperature:.2f}K")
            
            if rate > CRITICAL_MASS_THRESHOLD:
                print(f"System became critical at time {self.time:.4f}s")
                break
            
            # Adaptive time step
            time_step = min(initial_time_step, 1 / (rate + 1))

    def spatial_reaction_distribution(self):
        x = [event[1][0] for event in self.fission_events]
        y = [event[1][1] for event in self.fission_events]
        z = [event[1][2] for event in self.fission_events]
        return x, y, z

def plot_results(model):
    fig = plt.figure(figsize=(20, 15))
    
    # Reaction rate plot
    ax1 = fig.add_subplot(221)
    ax1.plot(model.time_steps, model.reaction_rates, label='Reaction Rate')
    ax1.axhline(y=CRITICAL_MASS_THRESHOLD, color='r', linestyle='--', label='Critical Mass Threshold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Reaction Rate')
    ax1.legend()
    
    # 3D fission events plot
    ax2 = fig.add_subplot(222, projection='3d')
    if model.fission_events:
        fission_events = np.array([event[1] for event in model.fission_events])  # event[1] is the position
        temperatures = np.array([event[2] for event in model.fission_events])  # event[2] is the temperature
        scatter = ax2.scatter(fission_events[:, 0], fission_events[:, 1], fission_events[:, 2],
                              c=temperatures, cmap='hot', s=10)
        fig.colorbar(scatter, label='Temperature (K)')
    else:
        ax2.text(0.5, 0.5, 0.5, "No fission events recorded", ha='center', va='center')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Fission Event Locations and Temperatures')
    
    # Neutron energy distribution
    ax3 = fig.add_subplot(223)
    energies = [n.energy for n in model.neutrons]
    if energies:
        sns.histplot(energies, kde=True, ax=ax3)
    else:
        ax3.text(0.5, 0.5, "No neutrons present", ha='center', va='center')
    ax3.set_xlabel('Neutron Energy (eV)')
    ax3.set_ylabel('Count')
    ax3.set_title('Neutron Energy Distribution')
    
    # Temperature evolution
    ax4 = fig.add_subplot(224)
    if model.fission_events:
        times = [event[0] for event in model.fission_events]
        temperatures = [event[2] for event in model.fission_events]
        ax4.plot(times, temperatures)
    else:
        ax4.text(0.5, 0.5, "No temperature data available", ha='center', va='center')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Temperature (K)')
    ax4.set_title('Temperature Evolution')
    
    plt.tight_layout()
    plt.show()

def analyze_results(model):
    if not model.fission_events:
        print("No fission events occurred during the simulation.")
        return

    # Clustering analysis of fission events
    fission_events = np.array([event[0] for event in model.fission_events])
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(fission_events)
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    print(f"Number of fission clusters: {n_clusters}")

    # Neutron lifecycle analysis
    if model.neutrons:
        lifecycle_lengths = [n.age for n in model.neutrons]
        avg_lifecycle = np.mean(lifecycle_lengths)
        print(f"Average neutron lifecycle length: {avg_lifecycle:.2f} steps")
    else:
        print("No neutrons present at the end of the simulation.")

    # Reactivity calculation
    if model.reaction_rates:
        k_eff = len(model.neutrons) / model.reaction_rates[-1] if model.reaction_rates[-1] > 0 else 0
        reactivity = (k_eff - 1) / k_eff if k_eff != 0 else float('inf')
        print(f"Estimated k_eff: {k_eff:.4f}")
        print(f"Reactivity: {reactivity:.4f}")
    else:
        print("No reaction rate data available for reactivity calculation.")


def monte_carlo_simulation(num_neutrons, material, geometry, total_time, initial_time_step):
    model = NuclearFissionModel(material, geometry)
    model.initialize_neutrons(num_neutrons)
    model.criticality_analysis(total_time, initial_time_step)
    plot_results(model)
    analyze_results(model)

if __name__ == "__main__":
    isotopes = [
        Isotope('U235', density=19.1, abundance=0.007),  # Reduced to 0.7% enrichment
        Isotope('U238', density=19.1, abundance=0.993),
        Isotope('Pu239', density=0.1, abundance=0.0001),
        Isotope('Pu241', density=0.01, abundance=0.00001),
        Isotope('Th232', density=0.01, abundance=0.00001)
    ]
    material = Material(isotopes, temperature=300)
    geometry = SphericalShellGeometry(inner_radius=0, outer_radius=10)
    monte_carlo_simulation(num_neutrons=10000, material=material, geometry=geometry, total_time=1.0, initial_time_step=1e-6)
