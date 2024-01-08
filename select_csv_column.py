import csv

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'results.csv'

with open(file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # Print the first, second, and last column of each row
        print(row[0], row[1], row[-1])

