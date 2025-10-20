# convert_to_single_label.py
input_file = "English.txt"
output_file = "English_test_single.txt"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        parts = line.strip().split()
        if len(parts) != 3:
            continue  # skip malformed lines
        _, voice_path, face_path = parts

        # Extract numeric IDs from filenames
        voice_id = voice_path.split("/")[-1].split(".")[0]
        face_id = face_path.split("/")[-1].split(".")[0]

        label = 1 if voice_id == face_id else 0
        f_out.write(f"{label}\n")

print("✅ Saved single-label format to", output_file)



# convert_to_double_label.py
input_file = "English.txt"
output_file = "English_test_double.txt"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        pair_id, voice_path, face_path = parts

        voice_id = voice_path.split("/")[-1].split(".")[0]
        face_id = face_path.split("/")[-1].split(".")[0]

        label = 1 if voice_id == face_id else 0
        f_out.write(f"{pair_id} {label}\n")

print("✅ Saved double-label format to", output_file)



import random

input_file = "English.txt"
single_output = "English_test_single.txt"
double_output = "English_test_double.txt"

# Optional: set a seed for reproducibility
random.seed(42)

with open(input_file, "r") as f_in, \
     open(single_output, "w") as f_single, \
     open(double_output, "w") as f_double:

    for line in f_in:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        pair_id, voice_path, face_path = parts

        # Random 0 or 1 label
        label = random.randint(0, 1)

        # Single-label format: just the label
        f_single.write(f"{label}\n")

        # Double-label format: pair ID and label
        f_double.write(f"{pair_id} {label}\n")

print("✅ Random single-label file saved to:", single_output)
print("✅ Random double-label file saved to:", double_output)