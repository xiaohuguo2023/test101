import csv

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'results.csv'

def create_markdown_table(file_path):
    table = ""
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Assuming the first row is the header
        # Create the header row for the markdown table
        table += "| " + " | ".join([headers[0], headers[1], headers[-1]]) + " |\n"
        # Create the separator row
        table += "|---|---|---|\n"
        # Add each data row
        for row in reader:
            table += "| " + " | ".join([row[0], row[1], row[-1]]) + " |\n"
    return table

# Call the function and print the markdown table
print(create_markdown_table(file_path))

