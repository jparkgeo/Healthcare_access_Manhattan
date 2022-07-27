import math
import numpy as np
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, Polygon, MultiPolygon
from tqdm import tqdm
import geopandas as gpd
import os

PWD = os.path.dirname(os.path.realpath(__file__))

def calculate_azimuth(x1, y1, x2, y2):
    d_x = x2 - x1  # Delta of x coordinates
    d_y = y2 - y1

    temp_azimuth = math.degrees(math.atan2(d_x, d_y))

    if temp_azimuth > 0:
        return temp_azimuth
    else:
        return temp_azimuth + 360


def calculate_weighted_mean(means, stds):
    return sum(means * 1 / stds) / sum(1 / stds)


def assign_speed_based_on_azimuth(network, u, v, match_df, uber_osm_edges):
    # The azimuth of current edges in the loop
    current_azimuth = round(calculate_azimuth(network.nodes[u]['x'], network.nodes[u]['y'],
                                              network.nodes[v]['x'], network.nodes[v]['y']), -1)

    match_df = match_df.merge(uber_osm_edges,
                              left_on=['osm_start_node_id', 'osm_end_node_id'],
                              right_on=['osmstartnodeid', 'osmendnodeid']
                              )

    if match_df.shape[0] > 0:
        match_df['azimuth'] = match_df.apply(lambda x: round(calculate_azimuth(x.geometry.coords[:][0][0],
                                                                               x.geometry.coords[:][0][1],
                                                                               x.geometry.coords[:][-1][0],
                                                                               x.geometry.coords[:][-1][1]
                                                                               ), -1)
                                             , axis=1)

        edges_of_interest = match_df.loc[match_df['azimuth'].isin([i + current_azimuth for i in range(-20, 30, 10)])]

        if edges_of_interest.shape[0] > 0:
            return calculate_weighted_mean(edges_of_interest['speed_mph_mean'], edges_of_interest['speed_mph_stddev'])
        else:
            return np.nan
    else:
        return np.nan


def remove_unnecessary_nodes(network):
    _nodes_removed = len([n for (n, deg) in network.out_degree() if deg == 0])
    network.remove_nodes_from([n for (n, deg) in network.out_degree() if deg == 0])
    for component in list(nx.strongly_connected_components(network)):
        if len(component) < 100:
            for node in component:
                _nodes_removed += 1
                network.remove_node(node)

    print("Removed {} nodes ({:2.4f}%) from the OSMNX network".format(_nodes_removed, _nodes_removed / float(
        network.number_of_nodes())))
    print("Number of nodes: {}".format(network.number_of_nodes()))
    print("Number of edges: {}".format(network.number_of_edges()))

    return network


def remove_unnecessary_edges(network, road_types):
    # Remove edges that car cannot travel; retrieved based on 'highway' attribute
    _edges_removed = 0
    for u, v, data in network.copy().edges(data=True):
        if 'highway' in data.keys():
            if type(data['highway']) == list:
                highway_type = data['highway'][0]
            else:
                highway_type = data['highway']

            # ['bridleway', 'access']
            if highway_type in road_types:  # Car cannot travel on these types of road
                network.remove_edge(u, v)
                _edges_removed += 1

    print("Removed {} edges ({:2.4f}%) from the OSMNX network".format(_edges_removed, _edges_removed /
                                                                      float(network.number_of_edges())))

    return network


def assign_max_speed_with_highway_type(network):
    # Maxspeed here was obtained from the mode of `maxspeed` attribute per `highway` type
    max_speed_per_type = {'motorway': 50,
                          'motorway_link': 30,
                          'trunk': 35,
                          'trunk_link': 25,
                          'primary': 25,
                          'primary_link': 25,
                          'secondary': 25,
                          'secondary_link': 25,
                          'tertiary': 25,
                          'tertiary_link': 25,
                          'residential': 25,
                          'living_street': 25,
                          'unclassified': 25,
                          'road': 25,
                          'track': 25
                          }

    for u, v, data in network.edges(data=True):
        # Assign the maximum speed of edges based on either 'maxspeed' or 'highway' attributes
        if 'maxspeed' in data.keys():
            if type(data['maxspeed']) == list:
                temp_speed = data['maxspeed'][0]  # extract only the first entry if there are many
            else:
                temp_speed = data['maxspeed']

            if type(temp_speed) == str:
                temp_speed = temp_speed.split(' ')[0]  # Extract only the number
                temp_speed = int(temp_speed)
        else:
            if 'highway' in data.keys():
                # if the variable is a list, grab just the first one.
                if type(data['highway']) == list:
                    road_type = data['highway'][0]
                else:
                    road_type = data['highway']

                # If the maximum speed of the road_type is predefined.
                if road_type in max_speed_per_type.keys():
                    temp_speed = max_speed_per_type[road_type]
                else:  # If not defined, just use 20 mph.
                    temp_speed = 20
            else:
                temp_speed = 20

        data['maxspeed'] = temp_speed  # Assign back to the original entry

    return network


def join_uber_data_with_osm_edges(network, uber_hour_df, uber_osm_gdf):

    for u, v, data in network.edges(data=True):
        # When travel time records in uber dataset match with OSM based on origin (u) and destination (v),
        # we update `uber_speed` of network (G) based on the uber records.
        match_ori_dest = uber_hour_df.loc[(uber_hour_df['osm_start_node_id'] == u) &
                                          (uber_hour_df['osm_end_node_id'] == v)
                                          ]
        # If there are matching records,
        # we average the travel time of Uber with the inverse weights of standard deviation.
        # Higher standard deviation has less weight
        if match_ori_dest.shape[0] > 0:
            weighted_mean = calculate_weighted_mean(match_ori_dest['speed_mph_mean'].values,
                                                    match_ori_dest['speed_mph_stddev'].values
                                                    )
            data['uber_speed'] = weighted_mean

        # If there is no matching records based on the origin (u) and destination (v),
        # we now try to match with their osm_way_id.
        # Given the osm_way_id is assigned based on the road names (i.e., multiple edges per osm_way_id),
        # we need to match the direction of edges (based on azimuth of edges) to retrieve travel time properly.
        else:
            if type(data['osmid']) == int:
                match_osmid = uber_hour_df.loc[uber_hour_df['osm_way_id'] == data['osmid']]

                if match_osmid.shape[0] > 0:
                    weighted_mean = assign_speed_based_on_azimuth(network, u, v, match_osmid, uber_osm_gdf)
                    data['uber_speed'] = weighted_mean

            elif type(data['osmid']) == list:
                speed_list = []
                for temp_id in data['osmid']:
                    match_osmid = uber_hour_df.loc[uber_hour_df['osm_way_id'] == temp_id]

                    if match_osmid.shape[0] > 0:
                        weighted_mean = assign_speed_based_on_azimuth(network, u, v, match_osmid, uber_osm_gdf)
                        speed_list.append(weighted_mean)

                if speed_list:
                    data['uber_speed'] = sum(speed_list) / len(speed_list)

    return network


def calculate_travel_time_of_edges(network):
    for u, v, data in network.edges(data=True):

        if 'uber_speed' in data.keys():
            if not math.isnan(data['uber_speed']):
                data['travel_speed'] = data['uber_speed']
            else:
                data['travel_speed'] = data['maxspeed']

        else:
            data['travel_speed'] = data['maxspeed']

        data['travel_dist'] = float(data['travel_speed']) * 26.8223  # MPH * 1.6 * 1000 / 60; meter per minute
        data['travel_time'] = float(data['length'] / data['travel_dist'])

    return network


def update_length_of_edges(network):
    nodes, edges = ox.graph_to_gdfs(network, nodes=True, edges=True, node_geometry=True)
    edges.reset_index(inplace=True)
    print(edges.crs)

    for u, v, data in network.edges(data=True):
        edge_length = edges.loc[(edges['u'] == u) & (edges['v'] == v), 'geometry'].length.values[0]
        data['length'] = edge_length

    return network


def network_settings(network):
    for node, data in network.nodes(data=True):
        data['geometry'] = Point(data['x'], data['y'])

    return network


def find_nearest_osm(network, gdf):
    """
    # This function helps you to find the nearest OSM node from a given GeoDataFrame
    # If geom type is point, it will take it without modification, but
    # IF geom type is polygon or multipolygon, it will take its centroid to calculate the nearest element.

    Input:
    - network (NetworkX MultiDiGraph): Network Dataset obtained from OSMnx
    - gdf (GeoDataFrame): stores locations in its `geometry` column

    Output:
    - gdf (GeoDataFrame): will have `nearest_osm` column, which describes the nearest OSM node
                          that was computed based on its geometry column

    """
    # for idx, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
    for idx, row in gdf.iterrows():
        if row.geometry.geom_type == 'Point':
            nearest_osm = ox.distance.nearest_nodes(network,
                                                    X=row.geometry.x,
                                                    Y=row.geometry.y
                                                    )
        elif row.geometry.geom_type == 'Polygon' or row.geometry.geom_type == 'MultiPolygon':
            nearest_osm = ox.distance.nearest_nodes(network,
                                                    X=row.geometry.centroid.x,
                                                    Y=row.geometry.centroid.y
                                                    )
        else:
            print(row.geometry.geom_type)
            continue

        gdf.at[idx, 'nearest_osm'] = nearest_osm

    return gdf


def calculate_catchment_area(network, nearest_osm, minutes, distance_unit='travel_time'):
    catchments = gpd.GeoDataFrame()

    # Create convex hull for each travel time (minutes), respectively.
    for minute in minutes:
        access_nodes = nx.single_source_dijkstra_path_length(network, nearest_osm, minute, weight=distance_unit)
        convex_hull = gpd.GeoSeries(
            nx.get_node_attributes(network.subgraph(access_nodes), 'geometry')).unary_union.convex_hull

        if type(convex_hull) == Polygon or type(convex_hull) == MultiPolygon:
            catchments.at[minute, 'geometry'] = convex_hull

    catchments.reset_index(inplace=True)
    catchments.rename(columns={'index': 'minutes'}, inplace=True)

    # Calculate the differences between convex hulls which created in the previous section.
    catchments_ = catchments.copy(deep=True)

    for idx in catchments_.index:
        if idx != 0:
            current_catchment = catchments_.loc[[idx], 'geometry'].values[0]
            previous_catchments = catchments_.loc[0:idx - 1, 'geometry'].unary_union
            diff_catchment = current_catchment.difference(previous_catchments)

            if type(diff_catchment) == Polygon or type(diff_catchment) == MultiPolygon:
                catchments.at[idx, 'geometry'] = diff_catchment

    if catchments.shape[0] > 0:
        catchments = catchments.set_crs(epsg=32118)

    return catchments.copy(deep=True)


def E2SFCA_Step1(day, hour, supply_loc, supply_open, supply_weight, demand_loc, demand_weight, network, distant_decay):
    supply_loc_ = supply_loc.copy(deep=True)
    supply_loc_[f'ratio_{day}_h{hour}'] = np.nan

    for s_idx, s_row in tqdm(supply_loc_.iterrows(), total=supply_loc_.shape[0]):

        if s_row[supply_open] == '1':  # If the facility is open
            supply_ctmt_area = calculate_catchment_area(network, s_row['nearest_osm'], distant_decay.keys())

            ctmt_pops = 0.0
            for c_idx, c_row in supply_ctmt_area.iterrows():
                temp_pop = demand_loc.loc[demand_loc.geometry.centroid.within(c_row['geometry']), demand_weight].sum()
                ctmt_pops += temp_pop * distant_decay[c_row['minutes']]

            supply_loc_.at[s_idx, f'ratio_{day}_h{hour}'] = s_row[supply_weight] / ctmt_pops * 100000
        else:
            supply_loc_.at[s_idx, f'ratio_{day}_h{hour}'] = 0.0

    if supply_loc_.shape[0] > 0:
        supply_loc_ = supply_loc_[['place_id', 'api_name', 'lat', 'long', 'api_addr', 'geometry',
                                   'nearest_osm', f'{day}_hours', 'doc_count', f'ratio_{day}_h{hour}']]


    return supply_loc_


def E2SFCA_Step2(day, hour, supply_loc, demand_loc, ratio_var, network, distant_decay):
    demand_loc_ = demand_loc.copy(deep=True)
    demand_loc_[f'access_{day}_h{hour}'] = np.nan

    for d_idx, d_row in tqdm(demand_loc_.iterrows(), total=demand_loc_.shape[0]):
        demand_ctmt_area = calculate_catchment_area(network, d_row['nearest_osm'], distant_decay.keys())

        ctmt_ratio = 0.0
        for c_idx, c_row in demand_ctmt_area.iterrows():
            temp_ratio = supply_loc.loc[supply_loc.geometry.centroid.within(c_row['geometry']), ratio_var].sum()
            ctmt_ratio += temp_ratio * distant_decay[c_row['minutes']]

        demand_loc_.at[d_idx, f'access_{day}_h{hour}'] = ctmt_ratio

    if demand_loc_.shape[0] > 0:
        demand_loc_ = demand_loc_[['GEOID', 'geometry',  'nearest_osm', f'access_{day}_h{hour}']]

    return demand_loc_


def G2SFCA_Step1(day, hour, supply_loc, supply_open, supply_weight, demand_loc, demand_weight, network, thres_trvl_time=30):
    supply_loc_ = supply_loc.copy(deep=True)
    supply_loc_[f'ratio_{day}_h{hour}'] = np.nan

    for s_idx, s_row in tqdm(supply_loc_.iterrows(), total=supply_loc_.shape[0]):
        if s_row[supply_open] == '1':  # If the facility is open
            access_nodes = nx.single_source_dijkstra_path_length(network,
                                                                 s_row['nearest_osm'],
                                                                 thres_trvl_time,
                                                                 weight='travel_time')
            convex_hull = gpd.GeoSeries(nx.get_node_attributes
                                        (network.subgraph(access_nodes), 'geometry')
                                        ).unary_union.convex_hull
            supply_ctmt_area = demand_loc.loc[demand_loc.geometry.centroid.within(convex_hull)]

            ctmt_pops = 0.0
            for c_idx, c_row in supply_ctmt_area.iterrows():
                trvl_time_ = nx.dijkstra_path_length(network,
                                                     s_row['nearest_osm'],
                                                     c_row['nearest_osm'],
                                                     weight='travel_time')
                if trvl_time_ <= thres_trvl_time:
                    ctmt_pops += c_row[demand_weight] * gaussian(trvl_time_, thres_trvl_time)

            supply_loc_.at[s_idx, f'ratio_{day}_h{hour}'] = s_row[supply_weight] / ctmt_pops * 100000
        else:
            supply_loc_.at[s_idx, f'ratio_{day}_h{hour}'] = 0.0

    if supply_loc_.shape[0] > 0:
        supply_loc_ = supply_loc_[['place_id', 'api_name', 'lat', 'long', 'api_addr', 'geometry',
                                   'nearest_osm', f'{day}_hours', 'doc_count', f'ratio_{day}_h{hour}']]

    return supply_loc_


def G2SFCA_Step2(day, hour, supply_loc, demand_loc, ratio_var, network, thres_trvl_time=30):
    demand_loc_ = demand_loc.copy(deep=True)
    demand_loc_[f'access_{day}_h{hour}'] = np.nan

    for d_idx, d_row in tqdm(demand_loc_.iterrows(), total=demand_loc_.shape[0]):
        access_nodes = nx.single_source_dijkstra_path_length(network,
                                                             d_row['nearest_osm'],
                                                             thres_trvl_time,
                                                             weight='travel_time')
        convex_hull = gpd.GeoSeries(nx.get_node_attributes
                                    (network.subgraph(access_nodes), 'geometry')
                                    ).unary_union.convex_hull
        demand_ctmt_area = supply_loc.loc[supply_loc.geometry.centroid.within(convex_hull)]

        ctmt_ratio = 0.0
        for c_idx, c_row in demand_ctmt_area.iterrows():
            trvl_time_ = nx.dijkstra_path_length(network,
                                                 d_row['nearest_osm'],
                                                 c_row['nearest_osm'],
                                                 weight='travel_time')
            if trvl_time_ <= thres_trvl_time:
                ctmt_ratio += c_row[ratio_var] * gaussian(trvl_time_, thres_trvl_time)

        demand_loc_.at[d_idx, f'access_{day}_h{hour}'] = ctmt_ratio

    if demand_loc_.shape[0] > 0:
        demand_loc_ = demand_loc_[['GEOID', 'geometry',  'nearest_osm', f'access_{day}_h{hour}']]

    return demand_loc_


def gaussian(dij, d0):  # Gaussian function for distance decay
    if d0 >= dij:
        val = (math.exp(-1 / 2 * ((dij / d0) ** 2)) - math.exp(-1 / 2)) / (1 - math.exp(-1 / 2))
        return val
    else:
        return 0


def measure_access(day, hour, supply_loc, demand_loc):
    G_hour = ox.load_graphml(
        os.path.join(PWD, 'data', 'reference_data', 'mobility', f'nyc_completed_{day}_{hour}.graphml'))
    G_hour = remove_unnecessary_nodes(G_hour)
    G_hour = network_settings(G_hour)

    supply_loc = find_nearest_osm(G_hour, supply_loc)
    demand_loc = find_nearest_osm(G_hour, demand_loc)

    supply_open = f'{day}_h{hour}'
    supply_weight = 'doc_count'
    demand_weight = f'{day}_h{hour}'
    ratio_var = f'ratio_{day}_h{hour}'

    supply_demand_ratio = G2SFCA_Step1(day,
                                       hour,
                                       supply_loc,
                                       supply_open,
                                       supply_weight,
                                       demand_loc,
                                       demand_weight,
                                       G_hour
                                       )

    access = G2SFCA_Step2(day,
                          hour,
                          supply_demand_ratio,
                          demand_loc,
                          ratio_var,
                          G_hour
                          )

    # supply_demand_ratio.to_file(f'data/reference_data/access/G2SFCA_step1_{day}_h{hour}.geojson')
    # access.to_file(f'data/reference_data/access/G2SFCA_step2_{day}_h{hour}.geojson')

    # print(supply_demand_ratio.head(5))
    # print(access.head(5))

    return supply_demand_ratio, access


def measure_access_unpacker(args):
    return measure_access(*args)


def measure_access_E2SFCA(day, hour, supply_loc, demand_loc):
    G_hour = ox.load_graphml(os.path.join(PWD, 'data', 'reference_data', 'mobility', f'nyc_completed_{day}_{hour}.graphml'))
    G_hour = remove_unnecessary_nodes(G_hour)
    G_hour = network_settings(G_hour)

    supply_loc = find_nearest_osm(G_hour, supply_loc)
    demand_loc = find_nearest_osm(G_hour, demand_loc)

    # minutes = list(range(1, 31))
    # gaussian_value = [gaussian(i, 30) for i in minutes]
    # gaussian_decay = {minutes[i]: gaussian_value[i] for i in range(len(minutes))}
    gaussian_decay = {10: 1, 20: 0.68, 30: 0.22}

    supply_open = f'{day}_h{hour}'
    supply_weight = 'doc_count'
    demand_weight = f'{day}_h{hour}'
    ratio_var = f'ratio_{day}_h{hour}'

    supply_demand_ratio = E2SFCA_Step1(day,
                                       hour,
                                       supply_loc,
                                       supply_open,
                                       supply_weight,
                                       demand_loc,
                                       demand_weight,
                                       G_hour,
                                       gaussian_decay
                                       )

    access = E2SFCA_Step2(day,
                          hour,
                          supply_demand_ratio,
                          demand_loc,
                          ratio_var,
                          G_hour,
                          gaussian_decay
                          )

    # supply_demand_ratio.to_file(f'data/reference_data/access/E2SFCA_step1_{day}_h{hour}.geojson')
    # access.to_file(f'data/reference_data/access/E2SFCA_step2_{day}_h{hour}.geojson')

    return supply_demand_ratio, access



def measure_access_E2SFCA_unpacker(args):
    return measure_access_E2SFCA(*args)