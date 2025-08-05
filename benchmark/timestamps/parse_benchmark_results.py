import json
import argparse

def write_row(csv, iterable, flush=True):
    csv.write(",".join(map(str, iterable)))
    if flush:
        csv.write("\n")

def main(args):
    input_file = args.json_file
    output_file = input_file.removesuffix("json") + "csv"
    with open(input_file, "rt") as f:
        raw_data = json.load(f)

    timestamps_per_req = raw_data["token_timestamps"]
    with open(output_file, "wt") as csv:
        for req_id, timestamps in enumerate(timestamps_per_req):
            write_row(csv, [req_id, -1] + [t for t in timestamps])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("json_file", type=str, help="The path of the json file")
    args = parser.parse_args()
    main(args)