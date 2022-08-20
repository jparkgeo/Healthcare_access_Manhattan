import geopandas as gpd
import osmnx as ox
import multiprocessing as mp
import itertools
import utils
import os

### -------- GLOBAL VARIABLES -------- ###
# day = 'wd'  # Weekday (wd), Weekend (we)
days = ['wd', 'we']  # Weekday (wd), Weekend (we)
hours = list(range(24))
PROCESSORS = int(os.environ['SLURM_TASKS_PER_NODE'])
# PROCESSORS = 8
RESULTS_FOLDER = os.getenv('result_folder')
print(os.environ)
print(f"The results folder is: {RESULTS_FOLDER}")
print(f"The number of tasks on Slurm is : {os.environ['SLURM_TASKS_PER_NODE']}")

# # print(f"CPU Count per tasks from mp package: {mp.cpu_count()}")
# print(f'entire os environment {os.environ}')
# print(f"CPU Count per tasks through os package: {os.environ['SLURM_TASKS_PER_NODE']}")

PWD = os.path.dirname(os.path.realpath(__file__))
print(PWD)

'''
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

product_day_hour = list(itertools.product(days, hours))
days_ = [day for day, hour in product_day_hour]
hours_ = [hour for day, hour in product_day_hour]

# days_ = ['we' for i in range(24)]
# hours_ = list(range(24))
# print(days_, hours_)

if __name__ == "__main__":
    pool = mp.Pool(PROCESSORS)
    results = pool.map(utils.measure_access_unpacker,
                       zip(days_,
                           hours_,
                           itertools.repeat(general_doctors),
                           itertools.repeat(commute_pop)
                           )
                       )
    pool.close()

    # Save the measures of accessibility
    for idx in range(len(results)):
        results[idx][0].to_file(f"./results/access/T_G2SFCA_step1_{days_[idx]}_h{hours_[idx]}.geojson")
        results[idx][1].to_file(f"./results/access/T_G2SFCA_step2_{days_[idx]}_h{hours_[idx]}.geojson")
'''