def update_employee_name(filename, record_number, new_name):
  """
  Updates the name field for a specific record in a binary employee data file.

  Args:
      filename (str): Path to the binary file.
      record_number (int): The index of the record to update (0-based).
      new_name (str): The new name to write to the file.
  """

  # Define employee record size (assuming integer id and fixed-length name)
  record_size = 4 + 20  # 4 bytes for integer id, 20 bytes for fixed-length name

  try:
    with open(filename, 'rb+') as file:
      # Calculate byte offset for the record
      byte_offset = record_number * record_size

      # Seek to the record position
      file.seek(byte_offset)

      # Read the existing id (optional, for verification)
      id_data = file.read(4)
      id = int.from_bytes(id_data, byteorder='little')

      # Encode the new name with fixed length (truncate or pad with spaces)
      new_name_encoded = new_name[:20].encode() + b' ' * (20 - len(new_name))

      # Write the updated name data
      file.write(new_name_encoded)

      print(f"Employee record #{record_number} (ID: {id}) name updated successfully!")

  except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
  except Exception as e:
    print(f"An error occurred: {e}")

# Test for file does not exist
filename = 'non_existent_file.bin'
record_number = 2  # Update the 3rd record (0-based index)
new_name = 'Alice Smith'

update_employee_name(filename, record_number, new_name)