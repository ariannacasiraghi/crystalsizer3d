import argparse
import json
from pathlib import Path

# Build parser object to parse command-line arguments
parser = argparse.ArgumentParser()

# Define the command-line options this script accepts
parser.add_argument('--original-split', type=Path, required=True, help='Path to the original train_test_split_0.80.json')
parser.add_argument('--n-train', type=int, required=True, help='Number of train indices to keep')
parser.add_argument('--n-test', type=int, required=True, help='Number of test indices to keep')
parser.add_argument('--output-dir', type=Path, required=True, help='Folder where the new split file will be written')


# Read the command line
args = parser.parse_args()

# Load the original JSON file for the train/test split (it's a dictionary)
with open(args.original_split, 'r') as f:
    split_data = json.load(f)

# Get the two lists of indices
train_idxs = split_data['train_idxs']
test_idxs = split_data['test_idxs']

# Keep the first n of each list. The original lists are already in random order, so a plain slice yields an unbiased sample
sub_train_idxs = train_idxs[:args.n_train]
sub_test_idxs = test_idxs[:args.n_test]

# Check that the length of each list matches the corresponding command-line value
assert len(sub_train_idxs) == args.n_train, 'Subset of train indices does not match n_train'
assert len(sub_test_idxs) == args.n_test, 'Subset of test indices does not match n_test'

# Build a dictionary with the same four keys used in the original train_test_split_0.80.json
tt_split = {
    'train_test_split_target': 0.8,
    'train_test_split_actual': len(sub_train_idxs) / (len(sub_train_idxs) + len(sub_test_idxs)),
    'train_idxs': sub_train_idxs,
    'test_idxs': sub_test_idxs,
}

# Check that the two lists do not have common elements
assert not (set(sub_train_idxs) & set(sub_test_idxs)), 'Test and train indices should be different'

# Check that the new train/test indices are a subset of the original train/test indices
assert set(sub_train_idxs) <= set(train_idxs), 'New train indices are not a subset of original train indices'
assert set(sub_test_idxs) <= set(test_idxs), 'New test indices are not a subset of original test indices'

# Check that the output folder is not the same as that of the original split, otherwise it would overwrite the JSON file
# resolve() is needed for the symlink: it turns the paths into real physical paths
assert args.output_dir.resolve() != args.original_split.resolve().parent, 'Output folder should not match folder of the original split'

# Output path
tt_split_path = args.output_dir / 'train_test_split_0.80.json'

# Create the directory if it doesn't exist
args.output_dir.mkdir(parents=True, exist_ok=True)

# Write the file
with open(tt_split_path, 'x') as f: # x as an additional safety to prevent overwrite
    json.dump(tt_split, f, indent=4)

# Reload the written file and check that it is as expected
with open(tt_split_path, 'r') as f:
    sub_split_data = json.load(f)
    assert sub_split_data == tt_split, 'The split written in the file does not match the intended split'

# Print
print(f'The sizes of the train and test splits are {len(sub_train_idxs)} and {len(sub_test_idxs)}.')
print(f'The output path is {tt_split_path}.')