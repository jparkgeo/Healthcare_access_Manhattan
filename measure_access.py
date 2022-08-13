import geopandas as gpd
import osmnx as ox
import multiprocessing as mp
import itertools
import utils
import os


### -------- GLOBAL VARIABLES -------- ###
day = 'wd'  # Weekday (wd), Weekend (we)
RESULTS_FOLDER = os.getenv('result_folder')
# print(f"The results folder is: {RESULTS_FOLDER}")
# print(f"CPU Count per tasks from mp package: {mp.cpu_count()}")
# print(f"CPU Count per tasks through os package: {os.environ['SLURM_CPUS_PER_TASK']}")
#
PWD = os.path.dirname(os.path.realpath(__file__))
print(PWD)

### -------- MAIN CODE STARTS HERE -------- ###
# Load input files
general_doctors = gpd.read_file('./data/reference_data/general_physicians.geojson')
commute_pop = gpd.read_file('./data/reference_data/floating_population_commute.geojson')
commute_pop = commute_pop.loc[commute_pop['GEOID'].str.startswith('36085')]
general_doctors = general_doctors.loc[general_doctors['geometry'].within(commute_pop.geometry.unary_union)]

# print(commute_pop.shape[0])
# print(general_doctors.shape[0])

G = ox.load_graphml(os.path.join(PWD, 'data', 'reference_data', 'mobility', 'nyc_completed_wd_12.graphml'))
G = utils.remove_unnecessary_nodes(G)
G = utils.network_settings(G)

# Precompute necessary variables
general_doctors = utils.find_nearest_osm(G, general_doctors)
commute_pop = utils.find_nearest_osm(G, commute_pop)
focus_hours = list(range(0, 24, 1))


if __name__ == "__main__":
    pool = mp.Pool(24)
    results = pool.map(utils.measure_access_E2SFCA_unpacker,
                       zip(itertools.repeat(day),
                           focus_hours,
                           itertools.repeat(general_doctors),
                           itertools.repeat(commute_pop)
                           )
                       )
    pool.close()

    # Save the measures of accessibility
    for idx, hour in enumerate(focus_hours):
        results[idx][0].to_file(os.path.join(RESULTS_FOLDER, f'E2SFCA_step1_{day}_h{hour}.geojson'))
        results[idx][1].to_file(os.path.join(RESULTS_FOLDER, f'E2SFCA_step2_{day}_h{hour}.geojson'))



