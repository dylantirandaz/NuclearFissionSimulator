# Nuclear Fission Simulator

This is a Twitter bot that scrapes insider sales data from FinViz and posts sales over a specified amount on Twitter.

## Explanation
### Neutron Class
Represents a neutron with a position and direction.
Methods move and scatter handle movement and scattering of neutrons, respectively.
### NuclearFissionModel Class
Manages the simulation of the nuclear fission process.
initialize_neutrons initializes neutrons with random directions.
simulate_transport simulates neutron transport and scattering.
reaction_rate calculates the reaction rate based on neutron density and material density.
criticality_analysis performs a criticality analysis, simulating the transport and reaction rate over time steps, and determining if the system becomes critical.
### Monte Carlo Simulation
monte_carlo_simulation function initializes the model, performs the criticality analysis, and plots the reaction rates over time.
## USAGE
Clone / Copy the repo
mkdir nuclear_fission_simulation
cd nuclear_fission_simulation
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
### Run the script
python nuclear_fission_simulation.py


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
