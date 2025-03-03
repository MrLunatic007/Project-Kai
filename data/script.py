# Input and output file names
input_file_1 = 'data\\dataset_2.csv'  # First input CSV file
input_file_2 = 'data\\text.csv'         # Second input CSV file
output_file = 'data\\dataset.csv'          # Combined output CSV file

# Function to clean a row (remove quotes and all numbers, leaving empty where quotes were)
def clean_row(row):
    # Remove quotation marks and numbers
    cleaned = [''.join(char for char in field if char not in ['"', "'", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '--T::Z']) for field in row]
    return cleaned

try:
    # Open the output file in write mode once
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        # Process first input file
        try:
            with open(input_file_1, mode='r', newline='', encoding='utf-8') as infile1:
                reader = infile1.readlines()
                for row in reader:
                    # Split by comma, clean, and join back
                    fields = row.strip().split(',')
                    cleaned_row = clean_row(fields)
                    outfile.write(','.join(cleaned_row) + '\n')
        except FileNotFoundError:
            print(f"Error: {input_file_1} not found. Skipping it.")

        # Process second input file
        try:
            with open(input_file_2, mode='r', newline='', encoding='utf-8') as infile2:
                reader = infile2.readlines()
                for row in reader:
                    # Split by comma, clean, and join back
                    fields = row.strip().split(',')
                    cleaned_row = clean_row(fields)
                    outfile.write(','.join(cleaned_row) + '\n')
        except FileNotFoundError:
            print(f"Error: {input_file_2} not found. Skipping it.")

    print(f"Processed files without quotation marks or numbers and saved to {output_file}")

except Exception as e:
    print(f"An error occurred: {e}")