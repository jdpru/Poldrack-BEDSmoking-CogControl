#!/usr/bin/bash

# all_batch=$(ls /oak/stanford/groups/russpold/data/uh2/aim4/BIDS/derivatives/output/*lev1_output/batch_files/*.batch)

## TEMPORARY ALL BATCH TO BE ONLY MANIPULATION TO FIGURE OUT WHY MANIPULATION ISN'T RUNNING 3/16
# all_batch=$(ls /oak/stanford/groups/russpold/data/uh2/aim4/BIDS/derivatives/output/manipulationTask_lev1_output/batch_files/*.batch)
all_batch=$(find /oak/stanford/groups/russpold/data/uh2/aim4/BIDS/derivatives/output/manipulationTask_lev1_output/batch_files -type f -name "*.batch")

# Print the list of batch files
echo "List of batch files:"
echo "$all_batch"

for cur_batch in ${all_batch}
do
  echo "Submitting batch file: $cur_batch"
  sbatch ${cur_batch} || echo "Error submitting batch file: $cur_batch"
done

echo "all done"