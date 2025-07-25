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
import multiprocessing
import time


#constants
NEUTRONS_PER_FISSION = {
    'U235': 2.43, 'U238': 2.3, 'Pu239': 2.88, 'Pu241': 2.93, 'Th232': 2.45
}
MEAN_FREE_PATH = {
    'U235': 2.0, 'U238': 2.5, 'Pu239': 1.8, 'Pu241': 1.9, 'Th232': 2.2
}  #cm
FISSION_PROBABILITY = {
    'U235': 0.00085, 'U238': 0.00015, 'Pu239': 0.00087, 'Pu241': 0.00089, 'Th232': 0.0001
} 
NEUTRON_ABSORPTION_PROBABILITY = 0.02
HEAT_CAPACITY = 0.2  #J/(g·K)
COOLING_RATE = 1e-8  #K/s
CRITICAL_MASS_THRESHOLD = 1000000
MAX_NEUTRONS = 1000000
BOLTZMANN_CONSTANT = 8.617333262145e-5  #eV/K

def generate_realistic_cross_section_data():
    isotopes = ['U235', 'U238', 'Pu239', 'Pu241', 'Th232']
    energies = np.logspace(-5, 7, 1000)  #10^-5 to 10^7 eV
    data = []

    for isotope in isotopes:
        if isotope == 'U235':
            base_xs = 100 + 500 * np.exp(-energies / 0.0253)#Peak at thermal energy (0.0253 eV)
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
        self.temperature = max(self.temperature, 300)  

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
        self.cumulative_fissions = 0
        self.energy_deposition = 0
        self.time = 0
        self.fission_events = deque(maxlen=10000)  
        self.neutron_lifecycles = {}
        self.energy_deposition = 0
        self.neutron_flux = []
        self.power_output = []


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
        a, b = 0.988, 2.249  
        x = np.random.random()
        E = -a * np.log(np.random.random()) - b * np.log(np.random.random())
        return a * np.sinh(np.sqrt(b * E))


    def determine_interaction(self, neutron, isotope):
        cross_section = isotope.cross_section_func(neutron.energy)
        fission_prob = FISSION_PROBABILITY[isotope.name] * cross_section / np.max(cross_section_data['cross_section'])
        fission_prob *= 0.001  #feel free to adjust if u want to reduce fission probability

        rand = np.random.random()
        if rand < fission_prob:
            return 'fission'
        elif rand < fission_prob + NEUTRON_ABSORPTION_PROBABILITY:
            return 'absorption'
        else:
            return 'scatter'

    def calculate_neutron_flux(self):
        volume = 4/3 * np.pi * (self.geometry.outer_radius**3 - self.geometry.inner_radius**3)
        return len(self.neutrons) / volume

    def calculate_power_output(self, fissions):
        #assuming 200 MeV per fission
        return fissions * 200 * 1.60218e-13  #convert MeV to Joules

    def simulate_transport(self, time_step):
        new_neutrons = []
        fissions_this_step = 0
        for neutron in self.neutrons:
            isotope = self.material.select_isotope()
            distance = np.random.exponential(MEAN_FREE_PATH[isotope.name])
            neutron.move(distance * time_step)
            
            if not self.geometry.is_inside(neutron.position):
                continue  #neutron escaped

            interaction_type = self.determine_interaction(neutron, isotope)
            if interaction_type == 'fission':
                fissions_this_step += 1
                self.cumulative_fissions += 1
                self.fission_events.append((self.time, neutron.position, self.material.temperature))
                self.energy_deposition += 200e6 * 1.60218e-19  #convert eV to joules
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
        self.energy_deposition = 0  #Reset energy deposition 
        
        self.neutron_flux.append(self.calculate_neutron_flux())
        self.power_output.append(self.calculate_power_output(fissions_this_step))
        
        return fissions_this_step

    def criticality_analysis(self, total_time, initial_time_step):
        time_step = initial_time_step
        while self.time < total_time:
            fissions = self.simulate_transport(time_step)
            rate = self.reaction_rate()
            self.reaction_rates.append(rate)
            self.time_steps.append(self.time)
            
            print(f"Time: {self.time:.4f}s, Fissions: {fissions}, Neutrons: {len(self.neutrons)}, T={self.material.temperature:.2f}K")
            
            if rate > CRITICAL_MASS_THRESHOLD and self.cumulative_fissions > 100:  
                print(f"System became critical at time {self.time:.4f}s")
                break
            
            time_step = min(initial_time_step, 1 / (rate + 1))
            self.time += time_step

    def spatial_reaction_distribution(self):
        x = [event[1][0] for event in self.fission_events]
        y = [event[1][1] for event in self.fission_events]
        z = [event[1][2] for event in self.fission_events]
        return x, y, z


def parallel_monte_carlo(num_simulations, num_neutrons, material, geometry, total_time, initial_time_step):
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(monte_carlo_simulation, num_neutrons, material, geometry, total_time, initial_time_step)
                   for _ in range(num_simulations)]
        results = [future.result() for future in futures]
    return results

def plot_results(model):
    fig = plt.figure(figsize=(20, 15))

    ax1 = fig.add_subplot(221)
    ax1.plot(model.time_steps, model.reaction_rates, label='Reaction Rate')
    ax1.axhline(y=CRITICAL_MASS_THRESHOLD, color='r', linestyle='--', label='Critical Mass Threshold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Reaction Rate')
    ax1.legend()

    ax2 = fig.add_subplot(222, projection='3d')
    if model.fission_events:
        fission_events = np.array([event[1] for event in model.fission_events])  
        temperatures = np.array([event[2] for event in model.fission_events]) 
        scatter = ax2.scatter(fission_events[:, 0], fission_events[:, 1], fission_events[:, 2],
                              c=temperatures, cmap='hot', s=10)
        fig.colorbar(scatter, label='Temperature (K)')
    else:
        ax2.text(0.5, 0.5, 0.5, "No fission events recorded", ha='center', va='center')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Fission Event Locations and Temperatures')

    
    ax5 = fig.add_subplot(325)
    ax5.plot(model.time_steps, model.neutron_flux)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Neutron Flux (n/cm³)')
    ax5.set_title('Neutron Flux over Time')

    ax6 = fig.add_subplot(326)
    ax6.plot(model.time_steps, model.power_output)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Power Output (W)')
    ax6.set_title('Power Output over Time')

    plt.tight_layout()
    plt.show()

    ax3 = fig.add_subplot(223)
    energies = [n.energy for n in model.neutrons]
    if energies:
        sns.histplot(energies, kde=True, ax=ax3)
    else:
        ax3.text(0.5, 0.5, "No neutrons present", ha='center', va='center')
    ax3.set_xlabel('Neutron Energy (eV)')
    ax3.set_ylabel('Count')
    ax3.set_title('Neutron Energy Distribution')

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

    num_fissions = len(model.fission_events)
    print(f"Total number of fission events: {num_fissions}")

    if num_fissions > 1:
        fission_locations = np.array([event[1] for event in model.fission_events])
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(fission_locations)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        print(f"Number of fission clusters: {n_clusters}")
    else:
        print("Not enough fission events for clustering analysis.")

    if model.neutrons:
        lifecycle_lengths = [n.age for n in model.neutrons]
        avg_lifecycle = np.mean(lifecycle_lengths)
        print(f"Average neutron lifecycle length: {avg_lifecycle:.2f} steps")
    else:
        print("No neutrons present at the end of the simulation.")

    if model.reaction_rates:
        k_eff = len(model.neutrons) / model.reaction_rates[-1] if model.reaction_rates[-1] > 0 else 0
        reactivity = (k_eff - 1) / k_eff if k_eff != 0 else float('inf')
        print(f"Estimated k_eff: {k_eff:.4f}")
        print(f"Reactivity: {reactivity:.4f}")
    else:
        print("No reaction rate data available for reactivity calculation.")

    if model.fission_events:
        temperatures = [event[2] for event in model.fission_events]
        avg_temp = np.mean(temperatures)
        max_temp = np.max(temperatures)
        print(f"Average temperature during fission events: {avg_temp:.2f} K")
        print(f"Maximum temperature reached: {max_temp:.2f} K")
    
    if model.neutron_flux:
        avg_flux = np.mean(model.neutron_flux)
        max_flux = np.max(model.neutron_flux)
        print(f"Average neutron flux: {avg_flux:.2e} n/cm³")
        print(f"Peak neutron flux: {max_flux:.2e} n/cm³")
    
    if model.power_output:
        avg_power = np.mean(model.power_output)
        max_power = np.max(model.power_output)
        print(f"Average power output: {avg_power:.2e} W")
        print(f"Peak power output: {max_power:.2e} W")


def monte_carlo_simulation(num_neutrons, material, geometry, total_time, initial_time_step):
    model = NuclearFissionModel(material, geometry)
    model.initialize_neutrons(num_neutrons)
    model.criticality_analysis(total_time, initial_time_step)
    plot_results(model)
    analyze_results(model)
    return model


if __name__ == "__main__":
    isotopes = [
        Isotope('U235', density=19.1, abundance=0.03),  #6-23-24 i increased enrichment to 3%
        Isotope('U238', density=19.1, abundance=0.97),
        Isotope('Pu239', density=0.1, abundance=0.0001),
        Isotope('Pu241', density=0.01, abundance=0.00001),
        Isotope('Th232', density=0.01, abundance=0.00001)
    ]
    material = Material(isotopes, temperature=300)
    geometry = SphericalShellGeometry(inner_radius=0, outer_radius=20)  #6-23-24 i increased size
    #monte_carlo_simulation(num_neutrons=10000, material=material, geometry=geometry, total_time=1.0, initial_time_step=1e-6)

    start_time = time.time()
    num_simulations = 1
    results = parallel_monte_carlo(num_simulations, num_neutrons=100000, material=material, geometry=geometry, total_time=100.0, initial_time_step=1e-5)
    end_time = time.time()

    print(f"Total simulation time for {num_simulations} runs: {end_time - start_time:.2f} seconds")

    for i, model in enumerate(results):
        print(f"\nResults for Simulation {i+1}:")
        analyze_results(model)
        plot_results(model)
