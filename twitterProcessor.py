import os
from mpi4py import MPI
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from collections import Counter
import numpy as np
import math
import json
import timeit
import operator
import csv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_core = comm.Get_size()
start = timeit.default_timer()

# Read in dataframe of language abbr_codes and languages
lang_abbr = []
language = []
with open("language.csv", "r") as csvfile:
    reader_variable = csv.reader(csvfile, delimiter=",")
    for line in reader_variable:
        lang_abbr.append((line[0]))
        language.append((line[1]))

# Create grid_language collection dictionary
grid_list = ["D1", "C1", "B1", "A1", "D2", "C2", "B2", "A2", "D3", "C3", "B3", "A3", "D4", "C4", "B4", "A4"]
grid_lang_list = [[] for i in range(len(grid_list))]
grid_dict = dict(zip(grid_list, grid_lang_list))

# Read in grid polygons
filename = "sydGrid.json"
polygon_dict = {}
polygon_raw_lst = []
with open(filename, 'r') as f:
    objects = json.load(f)
    
    jsons = objects["features"]
    for j in jsons:
        polygon_dict[j["properties"]["id"]] = Polygon(j['geometry']['coordinates'][0])

sort_polygon_dict = dict(sorted(polygon_dict.items()))

polygon_raw_lst = []
for key in sort_polygon_dict: 
    polygon_raw_lst.append(sort_polygon_dict[key])

Polygon_lst = []
for polygons in np.reshape(polygon_raw_lst, (4, 4)): 
    Polygon_lst.append(polygons[::-1])
Polygon_lst =  np.reshape(np.array(Polygon_lst), (1, 16)).tolist()[0]
polygon_num = len(Polygon_lst)

# Divide the twitter file into approximate equal size parts
data_filepath = 'bigTwitter.json'
file_size = os.path.getsize(data_filepath)
core_file_size = math.ceil(file_size / num_core)

with open(data_filepath, 'r', encoding = "utf8") as f:
    rank_read_start = rank * core_file_size
    rank_read_end = (rank + 1) * (core_file_size)
    if rank_read_end > file_size:
        rank_read_end = file_size

    map_file = f

    # Allocate pointers to the start of the allocated file content in each core
    map_file.seek(rank_read_start)
    current_pointer_index = rank_read_start

    # Start to read through the allocated file in each core and never stop until it meets the end
    while current_pointer_index < rank_read_end:
        # Read each line from the divided file with readline function
        
        line_record = map_file.readline()

        # Update the current pointer position with tell()
        current_pointer_index = map_file.tell()

        # Decode the json record line
        decoded_line = line_record

        # Preprocess before converting string to dict with json.loads
        new_line = decoded_line.rstrip("]}")
        new_line = new_line.rstrip("\n")
        new_line = new_line.rstrip("\r")
        new_line = new_line.rstrip(",")

        # Convert string to dict for further extraction of data
        try:
            record_dict = json.loads(new_line)
            
            # Check the valid coordinates' information
            if (record_dict["doc"]["coordinates"] != None):
                long = record_dict["doc"]["coordinates"]["coordinates"][0]
                lati = record_dict["doc"]["coordinates"]["coordinates"][1]

                for polyg in range(polygon_num):
                    if Polygon_lst[polyg].contains(Point(long, lati)) or Polygon_lst[polyg].touches(Point(long, lati)):
                        
                        # Check valided language from the tweet and collect them in to dict
                        try:
                            grid_dict[grid_list[polyg]].append(lang_abbr.index(record_dict["doc"]['lang']))
                            break
                        except Exception as e:
                            pass

        except Exception as e:
            pass

f.close()

# Combine dictionaries from the output of different cores
grid_dict_lst = comm.gather(grid_dict, root=0)

if (rank == 0):
    total_grid_dict = dict()

    for dict_per_core in grid_dict_lst:
        for key in dict_per_core:
            if key in total_grid_dict: 
                total_grid_dict[key] += (dict_per_core[key])
            else:
                total_grid_dict[key] = dict_per_core[key]

    # Collect number count of each language used in Sydney
    grid_sort = sorted(total_grid_dict)
    grid_sorted = {key:total_grid_dict[key] for key in grid_sort}

    # Convert language codes to actual language names
    for key in grid_sorted: 
        languages = []
        for lang_code in grid_sorted[key]:
            languages.append(language[lang_code])
        grid_sorted[key] = languages

    languages_lst = []
    total_languages_lst = []
    top10_language_lst = []
    top10_freq_lst = []

    # Calculate frequencies of language and sort them
    for key in grid_sorted: 
        grid_sorted[key] = dict(Counter(grid_sorted[key]))
        
        # Sort languages by their usage number 
        grid_sorted[key] = dict(sorted(grid_sorted[key].items(), key=operator.itemgetter(1),reverse=True))

        # add number of types of languages 
        languages_lst.append(len(grid_sorted[key]))

        # add total number of tweets
        total_languages_lst.append(sum(grid_sorted[key].values()))

    # add top ten languages and their corresponding frequency
    for key in grid_sorted: 
        freq_lst = []
        top10_language_lst.append(list(grid_sorted[key].keys())[0:10])
        top10_freq_lst.append(list(grid_sorted[key].values())[0:10])

    # form a top 10 languages & their corresponding number of Tweets list
    top10_language_freq_lst = []

    for i in range(len(top10_language_lst)): 
        area_languages = []
        for j in range(len(top10_language_lst[i])):
            area_languages.append(str(top10_language_lst[i][j] + '-' + str(top10_freq_lst[i][j])))
        top10_language_freq_lst.append(area_languages)
    
    # Print the output
    print("{:<8} {:<17} {:<18}     {:<18}".format("Cell", "#Total Tweets", "#Number of Languages Used", "#Top 10 Languages & #Tweets"))
    for i in range(len(languages_lst)): 
        print("{:<8} {:<17} {:<29} {:<14}".format(list(grid_sorted.keys())[i], str(total_languages_lst[i]), str(languages_lst[i]), "(" + str(top10_language_freq_lst[i])[1:-1] + ")"))

    # Check the duration for running the code
    print('Time: ', timeit.default_timer() - start)
