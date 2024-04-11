import sys
from utils.process_utils import combine_profile_csv, get_unique_mapping, scale_csv
# from utils import combine_data_csv

def main():
    ## Generate the combined csv.

    ## TODO: wrap this in another file; 
    ## process the results of NCU.
    ## E.g. extract metrics, etc, do some data science in a jupyter.

    ## Then, join the two dataframes...

    # combine_profile_csv()

    # res = get_unique_mapping()
    # for key, l in res.items():
    #     if len(l) > 1:
    #         print(f"{key}: {l}")

    scale_csv()

if __name__ == "__main__":
    main()