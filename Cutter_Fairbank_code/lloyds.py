"""
NAMES: Caroline Cutter & Julia Fairbank

CSCI 0311 - Artificial Intelligence 

ASSIGNMENT: FINAL PROJECT
            Lloyds Algorithm with Different Initializations 
            

DUE: Monday, May 23


DESCRIPTION:

    This python code contains implementations of Lloyd's method with
    5 different initalization methods (MacQueen, Forgy, K-Means++,
    Naive Sharding and Cutter-Fairbank Method). When this code is run,
    it will run for 100 iterations for each initialization method
    and information about accuracy, iterations and time will be written
    to a CSV file called "lloyds_data.csv". The files needed are
    wine_data.txt and wine_data.csv. The second half of this code
    contains the graphing code for the wine clustering. 
    
    

************************************************************************************"""




# ____________________________ PROGRAM INITIALZATION   __________________________________


# IMPORTS

from sklearn.preprocessing import StandardScaler    # Normalize Variables
from sklearn.decomposition import PCA               # PCA transformation for Graphing
from scipy.spatial import distance                  # Euclidean distance 
import matplotlib.pyplot as plt                     # Plotting Clusters
plt.style.use("seaborn-dark")
from math import floor                              # For Naive Sharding
import pandas as pd                                 # For standardizing and graphing data
import numpy as np                                  # For multidimensional array opperations 
import random                                       # For random initialization 
import time                                         # Keep track of time 
import csv                                          # Reading CSV File 



# GLOBALS
infinity = float("inf")





# READ FILE
def read_wine(filename):
    '''
    Take in wine data with m = 14 features
     
    Args:
        filename: string name of the file to read
    
    Returns:
        A list of m-dimensional array of points 
        
    '''
    with open(filename, "r") as file:
              
        points = [] 
        
        for line in file:
            points.append([float(coord) for coord in line.strip().split()]) 
 
        return points






# ____________________________ LLOYD'S ALGORITHM  __________________________________



def lloyds(k, m, points, initialization):
    '''
    Implements lloyds clustering algorithm with different initialization methods
     
    Args:
        k: number of points 
        m: integer dimensional space
        points: list of points 
    
    Returns:
        the centers of the clusters 

    '''
    start_time = time.time()
    
    # INITIALIZATION 
    centers = []
    
    # Basic Initialization
    if initialization == 0:
        centers = macqueen_init(points, k)
        
    # Forgy Initialization 
    elif initialization == 1:
        centers = forgy_init(points, k)
     
     
    # K-Means ++ Initialization 
    elif initialization == 2:
        centers = kmeansplusplus_init(points, k)
        
    # Naive Sharding Initialization 
    elif initialization == 3:
        centers = naive_sharding_init(points, k)
        
    # Julia and Caroline's Initialization
    else:
        centers = cutter_fairbank_init(points, k)



    difference = infinity
    num_iterations = 0
    
    while difference > 0:
        
        num_iterations += 1
        
        # Assign points to a center
        indices = [closest_center(point, centers, k, m) for point in points]
        
        # Find new centers 
        new_centers = [cluster_to_center(index, indices, m, points) for index in range(k)]
        
        # Check if any points moved, if not distance = 0 and loop stops 
        difference = sum([distance.euclidean(new_centers[i], centers[i]) for i in range(k)] )
        
        # Assing new centers as the centers
        centers = new_centers[:]

    
    end_time = time.time()
    total_time = end_time - start_time
    
    return centers, num_iterations, total_time 
    
    
    
    
    
    
   
# ____________________________ INITIALIZATION METHODS __________________________________



def macqueen_init(points, k):
    '''
    Initialization methods that takes the first k points as centers
     
    Args:
        points: list of instances as coordinate points
        k: number of clusters 
    
    Returns:
        k inital centers
    '''
   
    centers = [] 
    for i in range(k):
        centers.append(points[i])
        
    return centers





def forgy_init(points, k):
    '''
    Initialization method that picks k random points as centers
    
     
    Args:
        points: list of instances as coordinate points
        k: number of clusters 
    
    Returns:
        k inital centers
    '''
   
    centers = [] 
    for i in range(k):
        centers.append(random.choice(points))
        
    return centers
    



def kmeansplusplus_init(points, k):
    '''
    Initialization method that chooses the first center randomly,
    then the rest of the centers are chosen to be the centers by
    calculating the closest centers for each point, then choosing
    the point with the max distance to closest center

    Args:
        points: list of instances as coordinate points
        k: number of clusters 
    
    Returns:
        k inital centers
    '''
    
    centers = []
    
    # 1. Choose first center as random point in the data
    centers.append(random.choice(points))
    
    # Compute the rest of the centers
    for remaining in range(k - 1):
        
        distances = []
        
        # For each point 
        for point in points:
            
            min_dist = infinity
            
            # Calculate closest center
            for center in centers:
                
                current_dist = distance.euclidean(point, center)
                min_dist = min(min_dist, current_dist)
                
            distances.append(min_dist)
        
        # Select data point with max distance to closest center
        distances = np.array(distances)
        next_center = points[np.argmax(distances)]
        centers.append(next_center)
        distances = []
        
    return centers




def naive_sharding_init(points, k):
    '''
    Initialization method that chooses new centers by sorting and sharding
    composite values of the instances into k peices. The shards values are then
    averaged and the mean values become the initial centers. 
    
    Args:
        points: list of instances as coordinate points
        k: number of clusters 
    
    Returns:
        k inital centers
        
    *** THIS ALGORITHM WAS ADAPTED FROM:
        https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html
    ***
    ''' 
    # Initialize
    n = np.shape(points)[1]
    m = np.shape(points)[0]
    centers = np.mat(np.zeros((k,n)))

    # Sum the rows, add sum column, sort based off sum
    composite = np.mat(np.sum(points, axis=1))
    points = np.append(composite.T, points, axis=1)
    points.sort(axis=0)

    # Step for sharding
    step = floor(m/k)

    # Vectorize function 
    vectorize = np.vectorize(mean)

    # Shard the matrix rows to get k groups, average the shard columns
    # These averages become the centers
    for j in range(k):
        if j == k - 1:
            centers[j:] = vectorize(np.sum(points[j*step:,1:], axis=0), step)
        else:
            centers[j:] = vectorize(np.sum(points[j*step:(j+1)*step,1:], axis=0), step)

    return centers


def mean(total, step):
    ''' helper function for naive sharding, returns mean'''
    return total/step




def cutter_fairbank_init(points, k):
    '''
    Initialization method that chooses new centers by sorting and sharding
    composite values of the instances into k peices. The shards values are then
    averaged and the mean values become the initial centers. 
    
    Args:
        points: list of instances as coordinate points
        k: number of clusters 
    
    Returns:
        k inital centers
        
    *** THIS ALGORITHM IS A MIX OF:
            - Naive Sharding Method (https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html)
            - A method created by Al-Daoud et al. (10.5281/zenodo.1334075) 
    ''' 
    
    # Initalize variables 
    centers = []
    n = len(points)      # num points 
    m = len(points[0])   # num features
    
    
    # Find the variances of each column 
    variances = [] 
    for i in range(m):
        col = [point[i] for point in points]
        col_variance = np.var(col)
        variances.append(col_variance)
            
    
    # Sort by the column with max varaince 
    max_col = np.argmax(variances)
    points = sorted(points, key = lambda x: x[max_col])
    points = np.asarray(points)
    

    # Divide column into k "shards" 
    clusters = np.split(points, k)

    
    # For each cluster, find the means of all the columns to create new center
    for i in range(k):
        
        cluster = clusters[i]
        center = [] 
        
        # Calculate means for each feature
        for column in range(m): 
            
            col_average = sum(point[column] for point in cluster) / n
            
            center.append(col_average)
            
        centers.append(center)
        
    centers = np.asarray(centers)
  
    return centers
        
        
  
        
    
     
    
    
    






# ____________________________ LLOYD'S HELPER METHODS  __________________________________


def closest_center(point, centers, k, m):
    '''
    Finds closest point from a point to centers
     
    Args:
        point: list of coordinates of a point
        centers: a list of points
        k: number of centers
        m: integer dimensional space
    
    Returns:
        list of the coordinates for the best point
    
    '''
    best_center = -1
    best_dist = float("inf")
    
    # Find the closest center by euclidean distance
    for i in range(k):
        dist = distance.euclidean(point, centers[i])
        
        if dist < best_dist:
            best_center = i
            best_dist = dist
            
    return best_center





def cluster_to_center(index, indices, m, points):
    '''
    Transforms clusters to centers
     
    Args:
        index: index of a point
        indices: indicies of centers
        m: integer dimensional space
        points: list of points 
    
    Returns:
        returns the max point of the clusters to make new centers

    '''  
    total = 0
    coords = np.zeros(m)

    # Iterate through all the points 
    for i in range(len(points)):
        
        if indices[i] == index:
            total += 1
            for j in range(m):
                coords[j] += points[i][j]
    
    # Return the new centers 
    max_point = [c/max(total,1) for c in coords]
       
    return max_point
        




# ____________________________ GENERAL HELPER METHODS  __________________________________
    
def relabel(unlabeled_wine, wines):
    '''
    This function re-assings the labels on the newly clustered groups
    so that the labels "1, 2, 3" are the same groups as the original
    wine data. This is so the accuracy can be computed by comparing
    pairwise if the labels match for wines and unlabeled wines
    
    Args:
        unlabeled_wines: newly clustered dataset with unsupervised learning labels
        wines: the original dataset with real labels
        
    Returns:
        unlabeled wines with the corrected labels and wines


    '''
    # Initialize the counters
    
    # GROUP 1
    group1_center1 = 0
    group1_center2 = 0
    group1_center3 = 0
        
    # GROUP 2
    group2_center1 = 0
    group2_center2 = 0
    group2_center3 = 0  

    # GROUP 3
    group3_center1 = 0
    group3_center2 = 0
    group3_center3 = 0
    

    # For each unlabeled wine 
    for i, wine in enumerate(unlabeled_wines):
        
        # GROUP 1
        # If wine group is 1, find the original wine's label
        if wine[0] == 1:
            
            if wines[i][0] == 1:
                group1_center1 += 1
                
            if wines[i][0] == 2:
                group1_center2 += 1
                
                
            if wines[i][0] == 3:
                group1_center3 += 1
        
        
        # GROUP 2
        # If wine group is 2, find the original wine's label
        if wine[0] == 2:
            if wines[i][0] == 1:
                group2_center1 += 1
                
            if wines[i][0] == 2:
                group2_center2 += 1
                
                
            if wines[i][0] == 3:
                group2_center3 += 1
        
        
        # GROUP 3
        # If wine group is 3, find the original wine's label
        if wine[0] == 3:
            
            if wines[i][0] == 1:
                group3_center1 += 1
                
            if wines[i][0] == 2:
                group3_center2 += 1
                
                
            if wines[i][0] == 3:
                group3_center3 += 1
        
        
    # Find the max majority original label for unlabeled wine groups  

    unlabeled_group1 = [group1_center1, group1_center2, group1_center3]
    label_group1 = np.argmax(unlabeled_group1) + 1
    
    unlabeled_group2 = [group2_center1, group2_center2, group2_center3]
    label_group2 = np.argmax(unlabeled_group2) + 1
    
    unlabeled_group3 = [group3_center1, group3_center2, group3_center3]
    label_group3 = np.argmax(unlabeled_group3) + 1
                
    

    # Use these majorities to fix the unlabeled wines labels to match original 
    for wine in unlabeled_wines:
        
        if wine[0] == 1:
            wine[0] = label_group1
            
        elif wine[0] == 2:
            wine[0] = label_group2
            
        else:
            wine[0] = label_group3
         
         
    # Return the corrected set 
    return unlabeled_wines, wines
    
    
    
    




# ____________________________ MAIN METHOD  __________________________________

if __name__ == "__main__":
    
    
    # ____________________________ CREATING DATA FILE   __________________________________
    
    
    # Read in the file 
    wines = read_wine("wine_data.txt")


    # Remove groups for standardizing
    groups = []
    for x in wines:
        groups = groups + [x.pop(0)]


    # Standardize the values
    wines = StandardScaler().fit_transform(wines)
    wines = wines.tolist()
    

    # Add groups back in 
    for i, wine in enumerate(wines):
        wines[i] = [groups[i]] + wine
   
   
    # ARGUMENTS 
    k = 3
    features = 13
    runs = 100
    header = ['Initialization Method', 'Accuracy', 'Iterations', 'Time']
    
    
    with open('lloyds_data.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        
        # write the header
        writer.writerow(header)
        
        
        # WRITE THE DATA - for each initialization method
        for init in range(5):
            
            for iterations in range(runs):
                
                # Remove labels from wines
                unlabeled_wines = [row[1:] for row in wines]
                
                # Run lloyds to find centers
                centers, iterations, total_time = lloyds(k, features, unlabeled_wines, init)
 
                # Label each with closest center
                for i, wine in enumerate(unlabeled_wines):
        
                    # Find distance to centers
                    center_distances = [distance.euclidean(wine, centers[0]),
                                        distance.euclidean(wine, centers[1]),
                                        distance.euclidean(wine, centers[2])]
        
    
                    # New label is min distance to center
                    group = np.argmin(center_distances) + 1
                    unlabeled_wines[i] = [group] + unlabeled_wines[i]
                    
                    
                # Relabel the wines so they have the same group as original dataset (for comparisons)
                unlabeled_wines, wines = relabel(unlabeled_wines, wines)
                
                
                # Calculate Accuracy & Iterations
                total = 0
                total_correct = 0
 
                for i, wine in enumerate(wines):
        
                    total += 1
                    if wine[0] == unlabeled_wines[i][0]:
                        total_correct += 1
                
                
                # Print Out
                accuracy = round((total_correct / total * 100), 2)
                
                print("Total: ", (total))
                print("Total Correct: ", (total_correct))
                print("Accuracy: ", (accuracy))
                print("Iterations: ", (iterations))
                print("Time: ", (total_time))
                    
                # Add to CSV data sheet
                if init == 0:
                    initialization = "MacQueen"
                if init == 1:
                    initialization = "Forgy"
                if init == 2:
                    initialization = "Kmeans++"
                if init == 3:
                    initialization = "Naive Sharding"
                if init == 4:
                    initialization = "Cutter Fairbank" 
                
                writer.writerow([initialization, accuracy, iterations, total_time])
     
     
     
     
     
               
    # ____________________________ GRAPHING DATA   __________________________________


    '''
    NOTE: learned to graph and standardize values from:
          https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
          Code was taken and modified from here as well
    '''
    
    
# GLOBALS
    features = ['group','Alcohol','Malic acid',
                'Ash','Alcalinity of ash','Magnesium',
                'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                'Proanthocyanins', 'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines', 'Proline' ]
    groups = [1, 2, 3]
    colors = ['#FF82AB', '#33A1C9', '#00C957']
    
    
    wines = pd.read_csv("wine_data.csv", names = features)
    
    
# GRAPH 1: Original Groupings
    
    # Standardize the data
    x = wines.loc[:, features].values
    y = wines.loc[:,['group']].values
    x = StandardScaler().fit_transform(x)
    
    # PCA Transformation (13-D to 2-D)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principal_wines = pd.DataFrame(data = principalComponents,
                  columns = ['PCA Feature 1', 'PCA Feature 2'])


    # Combine final dataframe
    final_wines = pd.concat([principal_wines, wines[['group']]], axis = 1)
    
    # Graph the data
    fig1 = plt.figure(figsize = (8,8))
    figure1= fig1.add_subplot(1,1,1)
    figure1.set_xlabel('PCA Feature 1', fontsize = 15)
    figure1.set_ylabel('PCA Feature 2', fontsize = 15)
    figure1.set_title('Original Groups of Wine', fontsize = 20)
    for groups, color in zip(groups, colors):
        indices = final_wines['group'] == groups
        figure1.scatter(final_wines.loc[indices, 'PCA Feature 1'],
                        final_wines.loc[indices, 'PCA Feature 2'],
                        c = color, s = 50)
    figure1.legend(["Group 1", "Group 2", "Group 3"])
    figure1.grid()

    #plt.show()

    
    
    
    
    
    
# GRAPH 2: Groupings with PCA 
    
    
    # Handle the data
    values = final_wines.values
    wines = values.tolist()
    unlabeled_wines = [row[:2] for row in wines]



    # RUN LLOYDS
    # Get the centers
    centers, iterations, total_time = lloyds(3, 2, unlabeled_wines, 0)
  
    # Label data with closest center
    for i, wine in enumerate(wines):
               
        center_distances = [distance.euclidean(wine[:-1], centers[0]),
                            distance.euclidean(wine[:-1], centers[1]),
                            distance.euclidean(wine[:-1], centers[2])]
        
        group = np.argmin(center_distances)
        group += 1                           # account for starting at index 0
        
        unlabeled_wines[i] = unlabeled_wines[i] + [group]
       
    
    # Turn unlabeled_wines into a data frame 
    final_unlabeled_wines = pd.DataFrame(unlabeled_wines, columns=['PCA Feature 1', 'PCA Feature 2', 'group'])
        
    groups = [1, 2, 3]
    
    # Graph the data
    fig2 = plt.figure(figsize = (8,8))
    figure2= fig2.add_subplot(1,1,1)
    figure2.set_xlabel('PCA Feature 1', fontsize = 15)
    figure2.set_ylabel('PCA Feature 2', fontsize = 15)
    figure2.set_title('Wine Groups with Standardized Lloyds', fontsize = 20)
    for groups, color in zip(groups, colors):
        indices = final_unlabeled_wines['group'] == groups
        figure2.scatter(final_unlabeled_wines.loc[indices, 'PCA Feature 1'],
                        final_unlabeled_wines.loc[indices, 'PCA Feature 2'],
                        c = color, s = 50)
    figure2.legend(["Group 1", "Group 2", "Group 3"])
    figure2.grid()

    #plt.show()
    
    
    
# Graph 3: Groupings without Standardizing the Values
    
    # Handle the data
    wines = read_wine("wine_data.txt")
    unlabeled_wines = [row[1:] for row in wines]
    
    # RUN LLOYDS
    centers, iterations, total_time = lloyds(3, 13, unlabeled_wines, 0)

    # Label each with closest center
    for i, wine in enumerate(unlabeled_wines):

        # Find distance to centers
        center_distances = [distance.euclidean(wine, centers[0]),
                            distance.euclidean(wine, centers[1]),
                            distance.euclidean(wine, centers[2])]


        # New label is min distance to center
        group = np.argmin(center_distances) + 1
        unlabeled_wines[i] = [group] + unlabeled_wines[i]
        
        

    wines = pd.DataFrame(unlabeled_wines, columns=features)
        
        
    # Standardize the data
    x = wines.loc[:, features].values
    y = wines.loc[:,['group']].values
    x = StandardScaler().fit_transform(x)
    
    # PCA Transformation (13-D to 2-D)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principal_wines = pd.DataFrame(data = principalComponents,
                  columns = ['PCA Feature 1', 'PCA Feature 2'])


    # Combine final dataframe
    final_wines = pd.concat([principal_wines, wines[['group']]], axis = 1)
    groups = [1, 2, 3]
    
    # Graph the data
    fig3 = plt.figure(figsize = (8,8))
    figure3= fig3.add_subplot(1,1,1)
    figure3.set_xlabel('PCA Feature 1', fontsize = 15)
    figure3.set_ylabel('PCA Feature 2', fontsize = 15)
    figure3.set_title('Wine Groups with Non-Standardized Lloyds', fontsize = 20)
    for groups, color in zip(groups, colors):
        indices = final_wines['group'] == groups
        figure3.scatter(final_wines.loc[indices, 'PCA Feature 1'],
                        final_wines.loc[indices, 'PCA Feature 2'],
                        c = color, s = 50)
    figure3.legend(["Group 1", "Group 2", "Group 3"])
    figure3.grid()

    #plt.show()
    
    
    
    
    
    
# PLOTS FOR INITIALIZATIONS 


# GRAPH TO PUT MARKERS ON
    
    wines = pd.read_csv("wine_data.csv", names = features)
    
    
    # Standardize the data
    x = wines.loc[:, features].values
    y = wines.loc[:,['group']].values
    x = StandardScaler().fit_transform(x)
    
    # PCA Transformation (13-D to 2-D)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principal_wines = pd.DataFrame(data = principalComponents,
                  columns = ['PCA Feature 1', 'PCA Feature 2'])


    # Combine final dataframe
    final_wines = pd.concat([principal_wines, wines[['group']]], axis = 1)
    groups = [1, 2, 3]
    
    # Graph the data
    fig4 = plt.figure(figsize = (8,8))
    figure4 = fig4.add_subplot(1,1,1)
    figure4.set_xlabel('PCA Feature 1', fontsize = 15)
    figure4.set_ylabel('PCA Feature 2', fontsize = 15)
    for groups in groups:
        indices = final_wines['group'] == groups
        figure4.scatter(final_wines.loc[indices, 'PCA Feature 1'],
                        final_wines.loc[indices, 'PCA Feature 2'],
                        color = "#1C86EE", s = 50)
    figure4.grid()




    # HANDLE THE DATA 
    wines = principalComponents
    #print("here")
    
    #print(wines)
    
    
    # DIFFERENT INIT METHODS
    #centers = macqueen_init(wines, 3)
    #centers = forgy_init(wines, 3)
    #centers = kmeansplusplus_init(wines, 3)
    #centers = naive_sharding_init(wines, 3)
    centers = cutter_fairbank_init(wines[1:], 3)

    # Get coords of the centers
    centers = pd.DataFrame(data = centers,
                  columns = ['PCA Feature 1', 'PCA Feature 2'])
    
    x = centers['PCA Feature 1'].to_list()
    y = centers['PCA Feature 2'].to_list()

    
    # GRAPH THE MARKERS 
    
    figure4.scatter(x, y, marker = "x", s=80, c = "red")
    #figure4.set_title('MacQueen Initialization', fontsize = 20)
    #figure4.set_title('Forgy Initialization', fontsize = 20)
    #figure4.set_title('K-Means++ Initialization', fontsize = 20)
    #figure4.set_title('Naive Sharding Initialization', fontsize = 20)
    figure4.set_title('Cutter-Fairbank Initialization', fontsize = 20)
    
    plt.show()
    

   
    
