# Nuclear Fission Simulation

This project simulates neutron transport and nuclear fission processes using a Monte Carlo method. The simulation tracks neutron movements, interactions, and fission events within a defined geometry and material.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Classes and Functions](#classes-and-functions)
- [Simulation Parameters](#simulation-parameters)
- [Plotting and Analysis](#plotting-and-analysis)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need Python 3 and the following packages:

- numpy
- matplotlib
- scipy
- pandas
- seaborn
- scikit-learn

You can install the required packages using pip:

```bash
pip install numpy matplotlib scipy pandas seaborn scikit-learn
```
## Usage
To run the simulation, execute the `nuclear_fission_simulation.py` file:

```bash
python nuclear_fission_simulation.py
```

## Classes and Functions

### `generate_realistic_cross_section_data()`
Generates synthetic cross-section data for different isotopes.

### `Isotope`
Represents an isotope with its properties and cross-section data.

### `Neutron`
Represents a neutron with its position, direction, and energy.

### `Material`
Represents a material composed of different isotopes and manages its temperature and density.

### `SphericalShellGeometry`
Defines a spherical shell geometry for the simulation space.

### `NuclearFissionModel`
Handles the neutron transport, interactions, and fission processes.

- `initialize_neutrons(num_neutrons)`: Initializes neutrons in the geometry.
- `simulate_transport(time_step)`: Simulates neutron transport and interactions for a given time step.
- `criticality_analysis(total_time, initial_time_step)`: Analyzes the system for criticality over a given total time.
- `spatial_reaction_distribution()`: Returns the spatial distribution of fission events.
- `plot_results(model)`: Plots the results of the simulation, including reaction rates, fission event locations, neutron energy distribution, and temperature evolution.
- `analyze_results(model)`: Analyzes the results of the simulation, including clustering of fission events, neutron lifecycle analysis, and reactivity calculation.
- `monte_carlo_simulation(num_neutrons, material, geometry, total_time, initial_time_step)`: Runs the Monte Carlo simulation with the specified parameters.

## Simulation Parameters

- `num_neutrons`: Initial number of neutrons.
- `material`: Material object containing isotopes and temperature.
- `geometry`: Geometry object defining the simulation space.
- `total_time`: Total time for the simulation.
- `initial_time_step`: Initial time step for the simulation.

## Plotting and Analysis

The simulation results are visualized and analyzed through various plots and statistical methods:

- **Reaction Rate Plot**: Shows the reaction rate over time.
- **3D Fission Events Plot**: Displays the locations and temperatures of fission events.
- **Neutron Energy Distribution**: Histogram of neutron energies.
- **Temperature Evolution**: Tracks the temperature changes over time.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
