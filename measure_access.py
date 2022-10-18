import geopandas as gpd
import osmnx as ox
import multiprocessing as mp
import itertools
import utils
import os

### -------- GLOBAL VARIABLES -------- ###
days = ['wd', 'we']  # Weekday (wd), Weekend (we)
hours = list(range(24))
PROCESSORS = int(os.environ['SLURM_CPUS_PER_TASK'])  # Number of processors to use defined by SLURM; will define number of processes to run in parallel
RESULTS_FOLDER = os.getenv('result_folder')  # Temporary folder to store results

print(f"The results folder is: {RESULTS_FOLDER}")
print(f"The number of tasks on Slurm is : {os.environ['SLURM_CPUS_PER_TASK']}")

PWD = os.path.dirname(os.path.realpath(__file__))  # Get the current working directory
print(PWD)

### -------- MAIN CODE STARTS HERE -------- ###
# Load input files
general_doctors = gpd.read_file('./data/reference_data/general_physicians.geojson')  # Supply for accessibility measurements
commute_pop = gpd.read_file('./data/reference_data/floating_population_commute.geojson')  # Demand for accessibiltiy measurements

G = ox.load_graphml(os.path.join(PWD, 'data', 'reference_data', 'mobility', 'nyc_completed_wd_12.graphml'))  # Mobility for accessibility measurements
G = utils.remove_unnecessary_nodes(G)
G = utils.network_settings(G)

# Precompute necessary variables
general_doctors = utils.find_nearest_osm(G, general_doctors)
commute_pop = utils.find_nearest_osm(G, commute_pop)

# Create a list of all combinations of days and hours
product_day_hour = list(itertools.product(days, hours))
days_ = [day for day, hour in product_day_hour]
hours_ = [hour for day, hour in product_day_hour]

if __name__ == "__main__":
    pool = mp.Pool(PROCESSORS)
    results = pool.map(utils.measure_access_unpacker,
                       zip(days_, # String for weekday (wd) or weekend (we)
                           hours_, # Int for hour
                           itertools.repeat(general_doctors), # GeoDataFrame of general physicians
                           itertools.repeat(commute_pop) # GeoDataFrame of commute-adjusted population
                           )
                       )
    pool.close()

    # Save the measures of accessibility to temporary result folder
    # The output will be downloaded to CyberGISX through Globus 
    for idx in range(len(results)):
        results[idx][0].to_file(os.path.join(RESULTS_FOLDER, f"G2SFCA_step1_{days_[idx]}_h{hours_[idx]}.geojson"))
        results[idx][1].to_file(os.path.join(RESULTS_FOLDER, f"G2SFCA_step2_{days_[idx]}_h{hours_[idx]}.geojson"))
