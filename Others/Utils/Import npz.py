import numpy as np
import matplotlib.pyplot as plt


# Load the .npz file
with np.load("Others/Results/test_multiscale_log_density.npz") as data:
    # 1. List all variable names
    keys = data.files
    print(f"Found arrays: {keys}\n")
    

    for i in range(len(keys)):
        # 2. Access a specific array by its key
        # (Replace 'array_name' with an actual key from data.files)
        key = keys[i] 
        my_array = data[key]
        #if my_array.ndim > 0 and my_array.size > 0:
            #print(f"--- first 5 elements of '{my_array[:5]}' ---")
        # 3. Show the data
        print(f"--- Data for '{key}' ---")
        print("Shape:", my_array.shape)
        #print("Data Type:", my_type.dtype if hasattr(my_array, 'dtype') else type(my_array))
        print("Contents:\n", my_array)
        print("-------------------------------")
    
    frame_ind = data[keys[4]]
    count = 1
    for i in range(1, len(frame_ind)):
        if frame_ind[i] < frame_ind[i - 1]:
            count +=1 
            print(f"----- index of the first of the video '{count}' ---- '{i}' ------ ")

    print("-------------------------------")
    print(count)
    