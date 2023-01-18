"""
NAME: Caroline Cutter and Julia Fairbank

CSCI 0311A - Artificial Intelligence 

ASSIGNMENT: Final Project
    

DUE: Monday, May 23, 2022


DESCRIPTION:

    This method turns the data from the CSV into a txt file.
    The file needed is wine_data.csv and the information is
    written to "wine_data.txt"

"""

# IMPORTS
import csv 


# MAIN

if __name__ == "__main__":

    file = open("wine_data.csv")
    
    csvreader = csv.reader(file)
    
    header = []
    header = next(csvreader)
    
    examples = []
    
    for example in csvreader:
        examples.append(example)
        
      
    file.close()

    
    file = open("wine_data.txt", "a")

    print(len(examples))
    
    # Add to file 
    for example in examples:
        file.write(' '.join([element for element in example]) + '\n')
    file.close()
    