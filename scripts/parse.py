import csv, os, sys
import argparse
import pandas as pd
import torch

def parse_linear_params(op_type, path):
    unique_rows = set()
    data = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            b = int(row["b"]) if op_type == "bmm" else 1
            m = int(row["m"])
            k = int(row["k"])
            n = int(row["n"])

            row_key = (op_type, b, m, k, n)

            if row_key not in unique_rows:
                unique_rows.add(row_key)
                data.append({
                    "batch_size": b,
                    "m": m,
                    "k": k,
                    "n": n
                })

    # Convert the list to a pandas DataFrame
    return pd.DataFrame(data)

def parse_sdpa_params(path):
    unique_rows = set()
    data = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # query_shape == key_shape == value_shape.
            query_shape = eval(row["query_shape"])
            value_shape = eval(row["value_shape"])

            # Extract batch size, sequence length, and number of heads
            batch_size = query_shape[0]
            num_heads = query_shape[1]
            s_q = query_shape[2]
            d_qk = query_shape[3]

            s_kv = value_shape[2]
            d_v = value_shape[3]

            row_key = (batch_size, num_heads, s_q, s_kv, d_qk, d_v)

            if row_key not in unique_rows:
                unique_rows.add(row_key)
                data.append({
                    "batch_size": batch_size,
                    "num_heads": num_heads,
                    "s_q": s_q,
                    "s_kv": s_kv,
                    "d_qk": d_qk,
                    "d_v": d_v
                })

    # Convert list to pandas DataFrame
    return pd.DataFrame(data)

def process_padding(padding, kernel_size, stride):
    """Handles padding input to interpret 'valid', 'same', or tuple."""
    if isinstance(padding, str):
        if padding.lower() == 'valid':
            return 0
        elif padding.lower() == 'same':
            # Calculate 'same' padding based on stride and kernel size
            return (kernel_size // 2, kernel_size // 2)
        else:
            raise ValueError(f"Unexpected padding string: {padding}")
    elif isinstance(padding, (tuple, list)) and len(padding) == 2:
        return tuple(padding)
    elif isinstance(padding, int):
        return padding
    else:
        raise ValueError(f"Unexpected padding format: {padding}")

def process_single_or_tuple(value):
    """Process a single integer or a tuple/list of integers."""
    try:
        parsed = eval(value)
        if isinstance(parsed, int):
            return (parsed, parsed)
        elif isinstance(parsed, (tuple, list)) and len(parsed) == 1:
            return (parsed[0], parsed[0])
        if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
            return tuple(parsed)
        raise ValueError
    except Exception as e:
        raise ValueError(f"Unexpected format: {value}") from e

def parse_conv_params(path):
    """Parses parameters for conv2d operations."""
    unique_rows = set()
    data = []

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            x_shape = eval(row["x_shape"])
            w_shape = eval(row["w_shape"])
            bias = bool(row["bias"]) if row["bias"] != "" else None
            stride = process_single_or_tuple(row["stride"])
            padding = eval(row["padding"])
            dilation = process_single_or_tuple(row["dilation"])
            transposed = eval(row["transposed"])

            batch_size = x_shape[0]
            in_channels = x_shape[1]
            try:
                iH, iW = x_shape[2], x_shape[3]
            except:
                print(f"Convolution was missing fourth dimension: {x_shape}")
                continue

            out_channels = w_shape[0]
            kH, kW = w_shape[2], w_shape[3]

            # w_shape[1] = in_channels / groups.
            groups = in_channels // w_shape[1] if w_shape[1] != 0 else 1

            row_key = (
                batch_size, in_channels, iH, iW, out_channels, groups,
                kH, kW, stride, tuple(padding), dilation, transposed, bias
            )

            if row_key not in unique_rows:
                unique_rows.add(row_key)
                data.append({
                    "batch_size": batch_size,
                    "in_channels": in_channels,
                    "iH": iH,
                    "iW": iW,
                    "out_channels": out_channels,
                    "groups": groups,
                    "kH": kH,
                    "kW": kW,
                    "stride": stride,
                    "padding": padding,
                    "dilation": dilation,
                    "transposed": transposed,
                    "bias": bias
                })

    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Process parameters.")
    parser.add_argument("--op_type", type=str, required=True, choices=["addmm", "bmm", "mm", "sdpea", "sdpfa", "conv"],
                        help="The operation type (e.g., addmm, bmm, mm).")
    parser.add_argument("--path", type=str, required=True, help="The path to the CSV file with the parameters.")
    parser.add_argument("--save_path", type=str, required=False, help="The directory in which to save the csv.")
    parser.add_argument("--overwrite", type=str, default="a", choices=["a", "w"], help="Overwrite the save file.")
    args = parser.parse_args()

    # Validate that both --save_path and --overwrite are provided together
    if (args.save_path and not args.overwrite) or (not args.save_path and args.overwrite):
        parser.error("--save_path and --overwrite must be used together.")

    csv_path = os.path.join(args.path, args.op_type + ".csv")

    if not os.path.isfile(csv_path):
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(0)

    if args.op_type in ("addmm", "bmm", "mm"):
        df = parse_linear_params(args.op_type, csv_path)
    elif args.op_type in ("sdpea", "sdpfa"):
        df = parse_sdpa_params(csv_path)
    elif args.op_type in ("conv"):
        df = parse_conv_params(csv_path)
    
    if args.save_path:
        out_csv = os.path.join(args.save_path, args.op_type + ".csv")
        header = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
        with open(out_csv, args.overwrite, newline='') as f:
            df.to_csv(f, index=False, mode=args.overwrite, header=header)
    else:
        print(df)


if __name__ == "__main__":
    main()
