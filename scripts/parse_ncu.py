import os, sys, csv
import pandas as pd

def get_metric(row):
    """
    Store the metric name with the value, since units might be different 
    for each metric (e.g. gbyte/s vs. mbyte/s).
    """
    units = "" if str(row['Metric Unit']) == 'nan' else str(row['Metric Unit']) 
    return row['Metric Value'] + "[[" + units + "]]"

def main():
    in_file, out_file = str(sys.argv[1]), str(sys.argv[2])

    # TODO: make more helper functions
    # TODO: update switch statement once we include convolution.
    # TODO: add FLOPs (post-processing).
    # TODO: ensure everything is on the same unit scale (post-processing).
    # If there are multiple kernels generated, we post-process using kernel_params
    # to combine.

    full_df = pd.read_csv(in_file)
    full_df['ID'] = full_df['ID'].astype(int)
    num_entries = full_df['ID'].max()

    # TODO: switch case to deal with linear, conv, etc.
    if len(sys.argv) == 8:
        input_params = [str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6]), str(sys.argv[7])]

    kernel_params = ""
    for param in input_params:
        kernel_params += f"{param}" + "."
    kernel_params = kernel_params[:-1] # Remove the trailing period.

    for i in range(num_entries):
        df = full_df[full_df['ID'] == i].reset_index()
        assert(len(df) > 0)

        device_header = ['Kernel Name', 'Context', 'Stream', 'Block Size', 'Grid Size', 'Device', 'CC']
        for param in device_header:
            assert(df[param].nunique() == 1)
        extra_params = [df[param][0] for param in device_header]

        # Get non-null metrics
        res_df = df[df['Metric Name'].notna()][['Metric Name', 'Metric Unit', 'Metric Value']]
        res_df['Value'] = res_df.apply(get_metric, axis=1)
        res_df = res_df[['Metric Name', 'Value']]
        res_df.sort_values(by='Metric Name', inplace=True)

        keys = list(res_df['Metric Name'])
        values = list(res_df['Value'])

        with open(out_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not os.path.isfile(out_file) and os.path.getsize(out_file) > 0:
                device_header = ['Kernel Name', 'Context', 'Stream', 'Block Size', 'Grid Size', 'Device', 'CC']
                # TODO: switch case to deal with linear, conv, etc.
                if len(sys.argv) == 8:
                    input_header = ['Precision', 'Inputs', 'Bias', 'Input Size', 'Output Size']

                writer.writerow(["Params"] + input_header + device_header + keys)

            result_row = [kernel_params] + input_params + extra_params + values
            # Append the new row to the CSV file
            writer.writerow(result_row)


if __name__ == "__main__":
    main()