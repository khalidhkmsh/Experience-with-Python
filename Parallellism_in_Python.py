# +
# # !pip install --upgrade pandas
# -

import pandas as pd
from datetime import datetime
from multiprocessing import Pool
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path

# ==================================
# INSTRUCTIONS
# ==================================
# 1. Replace all places where there is 'YOUR CODE HERE" with your own code
# 2. You can test and check the code in Jupyter Notebook but the submission should be a Python file

# +
# pd.read_csv("activity_log_raw.csv",nrows=1000)
# -

def get_date_range_by_chunking(large_csv):
    """
    In this function, the idea is to use pandas chunk feature.
    :param large_csv: Full path to activity_log_raw.csv
    :return:
    """
    # ======================================
    # EXPLORE THE DATA
    # ======================================
    # Read the first 100,000 rows in the dataset
    df_first_100k = pd.read_csv(large_csv, nrows=100000)## YOUR CODE HERE
    print(df_first_100k.head())
    # Identify the time column in the dataset
    str_time_col = "ACTIVITY_TIME"##YOUR CODE HERE

    # ============================================================
    # FIND THE FIRST [EARLIEST] AND LAST DATE IN THE WHOLE DATASET
    # BY USING CHUNKING
    # =============================================================
    # set chunk size to some number. You can play around with the chunk size to see its effect
    chunksize = 100000 ##YOUR CODE HERE

    # declare a list to hold the dates
    dates =[] ##YOUR CODE HERE
    with pd.read_csv(large_csv, nrows=10000, chunksize=chunksize) as reader: # USE ONLY CHUNKSIZE !!
        for chunk in reader:
            # convert the string to Python datetime object
            # add a new column to hold this datetime object. For instance,
            # you can call it "activ_time"
            time_col ="activ_time" ##YOUR CODE HERE
            chunk[time_col] = chunk[str_time_col].apply(lambda x: pd.to_datetime(x[:9]))
            chunk.sort_values(by=time_col, inplace=True)
            top_date = chunk.iloc[0][time_col]
            # Add the top_date to the list
            dates.append(top_date)#YOUR CODE HERE
            chunk.sort_values(by=time_col, ascending=False, inplace=True)
            bottom_date = chunk.iloc[0][time_col]
            # Add the bottom_date to the list
            dates.append(bottom_date)##YOUR CODE HERE

    # print(len(dates))
    # Find the earliest and last date by sorting the dates list we created above
    sorted_dates =sorted(dates) ##YOUR CODE HERE
    first = sorted_dates[0]##YOUR CODE HERE
    last =  sorted_dates[-1] ##YOUR CODE HERE
    print("First date is {} and the last date is {}".format(first, last))

    return first, last
    
********************************************* MARKED *************************

# +
# get_date_range_by_chunking(large_csv="activity_log_raw.csv")
# -

def quadratic_func(x, a): ## MARKED
    """
    Define the quadratic function like this: y = 2x^2 + a -1
    (read as y is equal to 2 x squared plus a minus 1)
    :param x:
    :return:
    """
    y =2*x**2+a-1 ##YOUR CODE HERE

    # return value of y
    return y##YOUR CODE HERE


# +
# a=3;x=2
# quadratic_func(x, a)
# -

def run_the_quad_func_without_multiprocessing(list_x, list_y):## MARKED
    """
    Run the quadratic function on a huge list of X and Ys without using parallelism
    :param list_x: List of xs
    :param list_y: List of ys
    :return:
    """
    # use list comprehension and zip fucntion to achieve below
    results = list(zip(list_x,list_y))## YOUR CODE HERE

    # return thee results
    return results##YOUR CODE HERE


# +
# list_x=[1,2,3,4,5];list_y=[1,3,5,7,9]
# result=run_the_quad_func_without_multiprocessing(list_x,list_y)
# print(result)
# for e in result:
#     out=quadratic_func(e[0], e[1])
#     print (out)
# -

def run_the_quad_func_with_multiprocessing(list_x, list_y, num_processors=5):## MARKED
    """
    Run the quadratic function with multiprocessing
    :param list_x: List of xs
    :param list_y: List of xs
    :param num_processors: Number of processors to use
    :return:
    """
    # Use the Pool method to initiate processors with number
    # of processors set to num_processors
    processors =Pool(num_processors) ##YOUR CODE HERE
    params =list(zip(list_x,list_y))## YOUR CODE HERE

    # Use the starmap method for Pool to run the processes
    # Like this: Pool.starmap(func_name, params)
    results =  processors.starmap(quadratic_func, params)##YOUR CODE HERE
    processors.close()
    return results


# +
# # params =list(zip(list_x,list_y))
# params = [i for i in zip([1,2,3,4,5],[1,3,5,7,9])]
# params
# -

def multiprocessing_vs_sequential_quadratic(list_len, out_plot, out_csv):## MARKED
    """
    Compare how
    :param list_len:
    :return:
    """
    # declare a list to hold the data
    data =[]## YOUR CODE HERE

    # create a for loop usinng range function and list_len arg  
    for i in range(list_len): ##YOUR CODE HERE:
        list_length = 10 ** i

        # Create lists using the range() function
        # and provided list length param
        x =  list(range(list_length))#YOUR CODE HERE
        y =  list(range(list_length))##YOUR CODE HERE

        start_time = datetime.now()
        # Call the function run_the_quad_func_without_multiprocessing below
        run_the_quad_func_without_multiprocessing(x,y)##YOUR CODE HERE
        end_time = datetime.now()
        time_taken_seq = (end_time - start_time).total_seconds()
        data.append({'ListLen': list_length, 'Type' : 'Parallel', 'TimeTaken': time_taken_seq})

        start_time = datetime.now()
        # Call the function run_the_quad_func_with_multiprocessing below
        run_the_quad_func_with_multiprocessing(x, y, num_processors=5)##YOUR CODE HERE
        end_time = datetime.now()
        time_taken_mult = (end_time - start_time).total_seconds()
        data.append({'ListLen': list_length, 'Type' : 'Sequential', 'TimeTaken': time_taken_mult})

    # Create a data frame using the data variable defined above
    df = pd.DataFrame(data)##YOUR CODE HERE
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='ListLen', y='TimeTaken', hue='Type')

    # Use plt.savefig() to save the plot
    plt.savefig(out_plot)##YOUR CODE HERE

    # Also save the pandas dataframe defined above to csv
    df.to_csv(out_csv)##YOUR CODE HERE


def get_num_uniq_users(csv_file, userid_col):# Good
    """
    A Helper function to help get the number of unique users
    :param csv_file: path to CSV file
    :param userid_col: Column for user ID
    :return:
    """
    # Read the CSV file using pandas
    df =pd.read_csv(csv_file) #YOUR CODE HERE

    # Use the nunique() method to get number of unique users
    num =df[userid_col].nunique()##YOUR CODE HERE

    return num


def get_tot_uniq_users_parallel(path_to_csv, num_processors):# Good
    """

    :param path_to_csv:
    :return:
    """
    # ==================================================
    # GET LIST OF ALL CSV FILES AND PUT IN A LIST
    # ===================================================
    # convert the string URL for path to a Path object for easier interaction
    start = datetime.now()
    path_to_csv = Path(path_to_csv)
    list_csv = [f for f in path_to_csv.iterdir() if f.suffix == '.csv']


    # ======================================================
    # USE MULTIPROCESSING TO GET UNIQUE USERS FROM ALL CSV'S
    # ======================================================
    # Create processors using Pool and num_processors
    processors =Pool(num_processors)## YOUR CODE HERE

    # Prepare parameters for the get_num_uniq_users() function
    user_id_col = ['user_id']*len(list_csv)     # List containing parameters for the user_id column
    params = [i for i in zip(list_csv, user_id_col)]  # combine the two lists
    # Run the function in parallel
    results =processors.starmap(get_num_uniq_users, params)## YOUR CODE HERE
    processors.close()

    # combine results to get the total
    tot_users =sum(results) ##YOUR CODE HERE
    end = datetime.now()
    time_taken = round((end - start).total_seconds(), 2)
    print('Total unique users: {:,} in {} seconds'.format(tot_users, time_taken))

    return tot_users


def get_tot_uniq_users_seq(path_to_csv, userid_col):# Good
    """

    :param path_to_csv:
    :return:
    """
    # ==================================================
    # GET LIST OF ALL CSV FILES AND PUT IN A LIST
    # ===================================================
    # convert the string URL for path to a Path object for easier interaction
    start = datetime.now()
    path_to_csv = Path(path_to_csv)
    list_csv = [f for f in path_to_csv.iterdir() if f.suffix == '.csv']

    tot_users = 0
    for csv in list_csv:
        # Read CSV into pandas dataframe
        df =pd.read_csv(csv) ##YOUR CODE HERE
        # Get unique number of users using nunique() and the column for user_id
        uniq = df[userid_col].nunique()##YOUR CODE HERE

        # Increment the total number of users
        tot_users +=uniq ##YOUR CODE HERE

    end = datetime.now()
    time_taken = round((end - start).total_seconds(), 2)
    print('Total unique users: {:,} in {} seconds'.format(tot_users, time_taken))

    return tot_users


# +
# import os
# print(os.getcwd())
# -

if __name__ == '__main__':
    # Question-1: Pandas chunks
    file ="/home/khalid/Desktop/Big_Data/activity_log_raw.csv"
    get_date_range_by_chunking(file)

    # Question-2: CPU bound parallelization
    out_plot ="/home/khalid/Desktop/Big_Data/Kha_fig.png"## ADD PATH TO WHERE PLOT WILL BE SAVED WITH PNG EXTENSION
    out_csv ="/home/khalid/Desktop/Big_Data/Kha_df_file.csv"## ADD PATH TO WHERE CSV WILL BE SAVED WITH csv EXTENSION
    multiprocessing_vs_sequential_quadratic(9, out_plot, out_csv)

    # =====================================================
    # QUESTION-3: CPU BOUND PARALLELIZATION WITH PANDAS
    # =====================================================
    fpath ="/home/khalid/Desktop/Big_Data/simulated_cdrs" ##PATH WHERE YOU HAVE FOLDER WITH MANY SMALL CSV FILES
    get_tot_uniq_users_parallel(fpath, 5)
    get_tot_uniq_users_seq(fpath, 'user_id')



# #### 


