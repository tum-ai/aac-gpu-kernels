import os
import json

# Configuration
KERNELS_DIR = '../../'  # Directory containing your .cl kernel files
TEST_MAPPING_FILE = 'kernel_test_mapping.json'

def generate_boilerplate_json():
    existing_mappings = {}
    if os.path.exists(TEST_MAPPING_FILE):
        with open(TEST_MAPPING_FILE, 'r') as f:
            try:
                existing_data = json.load(f)
                # Convert list of dicts to a dict for easier lookup by kernel path
                existing_mappings = {entry['kernel']: entry.get('test', '') for entry in existing_data}
            except json.JSONDecodeError:
                print(f"WARNING: '{TEST_MAPPING_FILE}' is corrupted or empty. Starting with an empty mapping.")

    new_mappings = []
    
    for filename in os.listdir(KERNELS_DIR):
        if filename.endswith('.cl'):
            kernel_path = os.path.join(KERNELS_DIR, filename)
            # Normalize path to use in JSON for consistency (e.g., remove KERNELS_DIR prefix if it's '.')
            relative_kernel_path = os.path.relpath(kernel_path, KERNELS_DIR) if KERNELS_DIR != '.' else filename

            test_name = existing_mappings.get(relative_kernel_path)
            if test_name is None: # New kernel, add with empty test
                test_name = ""
                print(f"Adding new kernel: {relative_kernel_path}")
            else: # Existing kernel, use its existing test name
                print(f"Keeping existing entry for kernel: {relative_kernel_path}")

            new_mappings.append({
                "kernel": relative_kernel_path,
                "test": test_name
            })
    
    # Sort the new mappings for consistent output
    new_mappings.sort(key=lambda x: x['kernel'])

    with open(TEST_MAPPING_FILE, 'w') as f:
        json.dump(new_mappings, f, indent=4)
    
    print(f"\nBoilerplate JSON generated/updated in '{TEST_MAPPING_FILE}'.")
    print("Please review and fill in the 'test' values for new entries.")

if __name__ == "__main__":
    generate_boilerplate_json()