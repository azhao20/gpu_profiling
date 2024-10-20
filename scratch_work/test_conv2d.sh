#!/bin/bash

# Define arrays
sizes=(32 64 128 224 336 448 512 784 1120) # iH == iW
transposed=(0 1)
group_sizes=(1 16 64 128 256 512 768 1024)
batch_sizes=(2 4 8 16 32)
kernel_sizes=(3 5 7)
dtypes=("16" "32" "b16")
strides=(1)
dilations=(1)
channel_sizes=(1 8 16 64 128 256 512 768 1024)

# Initialize counters
total_combinations=0
transposed_combinations=0
non_transposed_combinations=0
skip_count=0
non_skip_count=0

# Iterate over all hyperparameter combinations
for iH in "${sizes[@]}"; do
  iW=$iH  # iH == iW
  for t in "${transposed[@]}"; do
    for group_size in "${group_sizes[@]}"; do
      for batch_size in "${batch_sizes[@]}"; do
        for kH in "${kernel_sizes[@]}"; do
          for stride in "${strides[@]}"; do
            for dilation in "${dilations[@]}"; do
              for dtype in "${dtypes[@]}"; do
                kW=$kH

                # Apply skipping condition based on iH and kH (kW == kH)
                if (( kH > iH )); then
                  skip_count=$((skip_count + 1))
                  continue
                fi

                if [ "$t" -eq 1 ]; then
                  # Transposed case: input_channels == output_channels
                  for channels in "${channel_sizes[@]}"; do
                    # Apply skipping condition based on channels and group size
                    if (( channels % group_size != 0 )); then
                      skip_count=$((skip_count + 1))
                      continue
                    fi

                    # Valid combination
                    total_combinations=$((total_combinations + 1))
                    transposed_combinations=$((transposed_combinations + 1))
                    non_skip_count=$((non_skip_count + 1))
                  done
                else
                  # Non-transposed case: different input/output channels
                  for in_channels in "${channel_sizes[@]}"; do
                    for out_channels in "${channel_sizes[@]}"; do
                      # Apply skipping condition based on in_channels, out_channels, and group size
                      if (( in_channels % group_size != 0 || out_channels % group_size != 0 )); then
                        skip_count=$((skip_count + 1))
                        continue
                      fi

                      # Valid combination
                      total_combinations=$((total_combinations + 1))
                      non_transposed_combinations=$((non_transposed_combinations + 1))
                      non_skip_count=$((non_skip_count + 1))
                    done
                  done
                fi

              done
            done
          done
        done
      done
    done
  done
done

# Print final counts
echo "Total combinations: $total_combinations"
echo "Transposed combinations: $transposed_combinations"
echo "Non-transposed combinations: $non_transposed_combinations"
echo "Skipped combinations: $skip_count"
echo "Non-skipped combinations: $non_skip_count"
