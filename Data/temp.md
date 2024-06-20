Sure, here's a Python code example that demonstrates how to interact with CSV files. This code covers reading from a CSV file, writing to a CSV file, and appending data to an existing CSV file.

First, ensure you have the necessary library installed:

```bash
pip install pandas
```

Here is the Python code:

```python
import pandas as pd

# Function to read a CSV file
def read_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data read from {file_path}:")
        print(data)
        return data
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except pd.errors.EmptyDataError:
        print(f"The file {file_path} is empty.")
    except pd.errors.ParserError:
        print(f"There was an error parsing {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to write a DataFrame to a CSV file
def write_csv(data, file_path):
    try:
        data.to_csv(file_path, index=False)
        print(f"Data written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to {file_path}: {e}")

# Function to append data to an existing CSV file
def append_to_csv(new_data, file_path):
    try:
        existing_data = pd.read_csv(file_path)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_csv(file_path, index=False)
        print(f"Data appended to {file_path}")
    except FileNotFoundError:
        print(f"The file {file_path} does not exist. Creating a new file.")
        new_data.to_csv(file_path, index=False)
    except pd.errors.EmptyDataError:
        print(f"The file {file_path} is empty. Writing new data.")
        new_data.to_csv(file_path, index=False)
    except pd.errors.ParserError:
        print(f"There was an error parsing {file_path}. Writing new data.")
        new_data.to_csv(file_path, index=False)
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    # File path for the CSV file
    file_path = 'example.csv'

    # Reading from a CSV file
    data = read_csv(file_path)

    # Writing to a CSV file
    new_data = pd.DataFrame({
        'Name': ['Alice', 'Bob'],
        'Age': [24, 30]
    })
    write_csv(new_data, 'new_example.csv')

    # Appending data to an existing CSV file
    more_data = pd.DataFrame({
        'Name': ['Charlie', 'David'],
        'Age': [35, 28]
    })
    append_to_csv(more_data, 'new_example.csv')
```

This code includes:
1. **Reading from a CSV file:** `read_csv` function reads the data from the specified CSV file and handles possible errors.
2. **Writing to a CSV file:** `write_csv` function writes a pandas DataFrame to a CSV file.
3. **Appending data to an existing CSV file:** `append_to_csv` function appends new data to an existing CSV file, handling cases where the file does not exist or is empty.