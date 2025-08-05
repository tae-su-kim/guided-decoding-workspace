import argparse
from collections import defaultdict
from typing import List, Set, DefaultDict, Tuple, Iterable

import matplotlib.colors
import matplotlib.pyplot
from tqdm import tqdm

COLORS = [(1.0, 0.5, 0.5), (0.9375, 0.41015625, 0.41015625), (0.875, 0.328125, 0.328125), (0.8125, 0.25390625, 0.25390625), (0.75, 0.1875, 0.1875), (0.6875, 0.12890625, 0.12890625), (0.625, 0.078125, 0.078125), (0.5625, 0.03515625, 0.03515625), (1.0, 0.6875, 0.5), (0.9375, 0.60791015625, 0.41015625), (0.875, 0.533203125, 0.328125), (0.8125, 0.46337890625, 0.25390625), (0.75, 0.3984375, 0.1875), (0.6875, 0.33837890625, 0.12890625), (0.625, 0.283203125, 0.078125), (0.5625, 0.23291015625, 0.03515625), (1.0, 0.875, 0.5), (0.9375, 0.8056640625, 0.41015625), (0.875, 0.73828125, 0.328125), (0.8125, 0.6728515625, 0.25390625), (0.75, 0.609375, 0.1875), (0.6875, 0.5478515625, 0.12890625), (0.625, 0.48828125, 0.078125), (0.5625, 0.4306640625, 0.03515625), (0.9375, 1.0, 0.5), (0.87158203125, 0.9375, 0.41015625), (0.806640625, 0.875, 0.328125), (0.74267578125, 0.8125, 0.25390625), (0.6796875, 0.75, 0.1875), (0.61767578125, 0.6875, 0.12890625), (0.556640625, 0.625, 0.078125), (0.49658203125, 0.5625, 0.03515625), (0.75, 1.0, 0.5), (0.673828125, 0.9375, 0.41015625), (0.6015625, 0.875, 0.328125), (0.533203125, 0.8125, 0.25390625), (0.46875, 0.75, 0.1875), (0.408203125, 0.6875, 0.12890625), (0.3515625, 0.625, 0.078125), (0.298828125, 0.5625, 0.03515625), (0.5625, 1.0, 0.5), (0.47607421875, 0.9375, 0.41015625), (0.396484375, 0.875, 0.328125), (0.32373046875, 0.8125, 0.25390625), (0.2578125, 0.75, 0.1875), (0.19873046875, 0.6875, 0.12890625), (0.146484375, 0.625, 0.078125), (0.10107421875, 0.5625, 0.03515625), (0.5, 1.0, 0.625), (0.41015625, 0.9375, 0.5419921875), (0.328125, 0.875, 0.46484375), (0.25390625, 0.8125, 0.3935546875), (0.1875, 0.75, 0.328125), (0.12890625, 0.6875, 0.2685546875), (0.078125, 0.625, 0.21484375), (0.03515625, 0.5625, 0.1669921875), (0.5, 1.0, 0.8125), (0.41015625, 0.9375, 0.73974609375), (0.328125, 0.875, 0.669921875), (0.25390625, 0.8125, 0.60302734375), (0.1875, 0.75, 0.5390625), (0.12890625, 0.6875, 0.47802734375), (0.078125, 0.625, 0.419921875), (0.03515625, 0.5625, 0.36474609375), (0.5, 1.0, 1.0), (0.41015625, 0.9375, 0.9375), (0.328125, 0.875, 0.875), (0.25390625, 0.8125, 0.8125), (0.1875, 0.75, 0.75), (0.12890625, 0.6875, 0.6875), (0.078125, 0.625, 0.625), (0.03515625, 0.5625, 0.5625), (0.5, 0.8125, 1.0), (0.41015625, 0.73974609375, 0.9375), (0.328125, 0.669921875, 0.875), (0.25390625, 0.60302734375, 0.8125), (0.1875, 0.5390625, 0.75), (0.12890625, 0.47802734375, 0.6875), (0.078125, 0.419921875, 0.625), (0.03515625, 0.36474609375, 0.5625), (0.5, 0.625, 1.0), (0.41015625, 0.5419921875, 0.9375), (0.328125, 0.46484375, 0.875), (0.25390625, 0.3935546875, 0.8125), (0.1875, 0.328125, 0.75), (0.12890625, 0.2685546875, 0.6875), (0.078125, 0.21484375, 0.625), (0.03515625, 0.1669921875, 0.5625), (0.5625, 0.5, 1.0), (0.47607421875, 0.41015625, 0.9375), (0.396484375, 0.328125, 0.875), (0.32373046875, 0.25390625, 0.8125), (0.2578125, 0.1875, 0.75), (0.19873046875, 0.12890625, 0.6875), (0.146484375, 0.078125, 0.625), (0.10107421875, 0.03515625, 0.5625), (0.75, 0.5, 1.0), (0.673828125, 0.41015625, 0.9375), (0.6015625, 0.328125, 0.875), (0.533203125, 0.25390625, 0.8125), (0.46875, 0.1875, 0.75), (0.408203125, 0.12890625, 0.6875), (0.3515625, 0.078125, 0.625), (0.298828125, 0.03515625, 0.5625), (0.9375, 0.5, 1.0), (0.87158203125, 0.41015625, 0.9375), (0.806640625, 0.328125, 0.875), (0.74267578125, 0.25390625, 0.8125), (0.6796875, 0.1875, 0.75), (0.61767578125, 0.12890625, 0.6875), (0.556640625, 0.078125, 0.625), (0.49658203125, 0.03515625, 0.5625), (1.0, 0.5, 0.875), (0.9375, 0.41015625, 0.8056640625), (0.875, 0.328125, 0.73828125), (0.8125, 0.25390625, 0.6728515625), (0.75, 0.1875, 0.609375), (0.6875, 0.12890625, 0.5478515625), (0.625, 0.078125, 0.48828125), (0.5625, 0.03515625, 0.4306640625), (1.0, 0.5, 0.6875), (0.9375, 0.41015625, 0.60791015625), (0.875, 0.328125, 0.533203125), (0.8125, 0.25390625, 0.46337890625), (0.75, 0.1875, 0.3984375), (0.6875, 0.12890625, 0.33837890625), (0.625, 0.078125, 0.283203125), (0.5625, 0.03515625, 0.23291015625)]


POINTS = List[int]
PLOT_DATA = Tuple[POINTS, List[POINTS]]
def parse_data(file_path: str, num_requests: int=-1, num_steps: int=-1) -> PLOT_DATA:
    req_ids: Set[int] = set()
    points: DefaultDict[float, Set[int]] = defaultdict(set)

    # read data from file
    with open(file_path, "rt") as f:
        lines = f.readlines()
        with tqdm(total=len(lines), desc="reading data") as bar:
            for rid, line in enumerate(lines):
                cells = line.split(",")
                
                # rid = int(cells[0])
                cells = list(map(lambda x: float(x), cells))
                timestamps = cells
                req_ids.add(rid)

                for point in set(timestamps):
                    points[point].add(rid)

                bar.update(1)
           
    # convert data to stacked area plot format
    reqs_to_draw = set()
    num_requests = len(req_ids) if num_requests == -1 else num_requests
    num_steps_to_parse = len(points) if num_steps == -1 or len(points) < num_steps else num_steps
    x: POINTS = list(range(num_steps_to_parse))
    ys: List[POINTS] = [[0]*num_steps_to_parse for _ in range(max(req_ids) + 1)]
    steps_to_parse = enumerate(sorted(points.items(), key=lambda p: p[0])[:num_steps_to_parse])
    for step, (_, rids) in tqdm(steps_to_parse, desc="reformatting data"):
        for rid in rids:
            if len(reqs_to_draw) < num_requests:
                reqs_to_draw.add(rid)

        for rid in rids.intersection(reqs_to_draw):
            ys[rid][step] = 1

    return x, ys


def correction(x: POINTS, ys: List[POINTS]) -> PLOT_DATA:
    def get_activate_requests(step: int, ys: List[List[int]]) -> Set[int]:
        return set([req_id for req_id, y in enumerate(ys) if y[step]])
    
    iter = 1
    start = 0
    while True:
        i = start
        end = len(x) - 2
        updated = False
        with tqdm(total=end-start+1, desc=f"correcting errancies(attempt {iter})") as pbar:
            while i <= end:
                prev_reqs = get_activate_requests(i - 1, ys)
                steps_of_interest = [get_activate_requests(i + j, ys) for j in range(2)]

                detected = True
                for reqs in steps_of_interest:
                    if not prev_reqs.issuperset(reqs):
                        detected = False
                        break
                    
                    prev_reqs.difference_update(reqs)
                
                if not detected:
                    pbar.update(1)
                    i += 1
                    continue
                
                if not updated:
                    updated = True
                    start = i

                del x[i+1]
                for y in ys:
                    if y.pop(i+1):
                        y[i] = 1
                end -= 1
                pbar.update(2)
                i += 1

        if not updated:
            break

        iter += 1

    ys = [y for y in ys if any(y)]
    return x, ys


def analyze(x: POINTS, ys: List[POINTS], csv_path: str) -> None:
    def get_max_and_avg(iterable: POINTS) -> Tuple[int, float]:
        return max(iterable), (sum(iterable) / len(iterable))
    
    def count_preemption(y: POINTS, first: int, last: int) -> int:
        return "".join(map(str, y[first:last+1])).count("10")
    
    def write_row(csv, items: Iterable, flush=True) -> None:
        csv.write(",".join(map(str, items)))
        if flush:
            csv.write("\n")

    # general
    total_steps = len(x)
    total_active_steps = len([any(y[s] if s < len(y) else 0 for y in ys) for s in x])
    total_active_reqs = len(ys)

    # batch size
    bs_per_step: List[int] = [sum(y[step] if step < len(y) else 0 for y in ys) for step in x]
    max_bs, avg_bs = get_max_and_avg(bs_per_step)

    # active steps per requests
    active_steps_per_req = [sum(y) for y in ys]
    max_active_step, avg_active_step = get_max_and_avg(active_steps_per_req)

    # duration per requests
    first_steps_per_req = [y.index(1) if 1 in y else -1 for y in ys]
    last_steps_per_req = [len(y) - 1 - y[::-1].index(1) if 1 in y else -1 for y in ys]
    duration_per_req = list(map(int.__sub__, last_steps_per_req, first_steps_per_req))
    max_duration, avg_duration = get_max_and_avg(duration_per_req)

    # preemptions
    zipped = zip(ys, first_steps_per_req, last_steps_per_req)
    preemption_occurrence_per_req = [count_preemption(y, f, l) for y, f, l in zipped]
    total_preemption = sum(preemption_occurrence_per_req)
    max_preemption, avg_preemption = get_max_and_avg(preemption_occurrence_per_req)

    print("GENERAL:")
    print(f"\tTOTAL STEPS: {total_steps}")
    print(f"\tTOTAL ACTIVE STEPS: {total_active_steps}")
    print(f"\tTOTAL ACTIVE REQUESTS: {total_active_reqs}")

    print("BATCH SIZE:")
    print(f"\tMAX RUNNING BATCH SIZE: {max_bs}")
    print(f"\tAVG RUNNING BATCH SIZE: {avg_bs}")

    print("REQUEST DURATIONS:")
    print(f"\tMAX ACTIVE STEP: {max_active_step}")
    print(f"\tAVG ACTIVE STEP: {avg_active_step}")
    print(f"\tMAX DURATION(STEP TIL COMPLETION): {max_duration}")
    print(f"\tAVG DURATION(STEP TIL COMPLETION): {avg_duration}")

    print("PREEMPTIONS:")
    print(f"\tTOTAL PREEMPTION OCCURRENCE: {total_preemption}")
    print(f"\tMAX PREEMPTION PER REQUEST: {max_preemption}")
    print(f"\tAVG PREEMPTION PER REQUEST: {avg_preemption}")

    if csv_path:
        with open(csv_path, "wt") as csv:
            write_row(csv, ["total steps", "total active steps", "total active requests"])
            write_row(csv, [total_steps, total_active_steps, total_active_reqs])
            csv.write("\n")

            write_row(csv, ["", "batch size", "active steps", "duration", "preemption"])
            write_row(csv, ["max", max_bs, max_active_step, max_duration, max_preemption])
            write_row(csv, ["avg", avg_bs, avg_active_step, avg_duration, avg_preemption])
            csv.write("\n")

            csv.write("request-wise data:\n")
            write_row(csv, ["num active steps", "first step", "last step", "duration", "num preemption"])
            zipped = zip(active_steps_per_req, first_steps_per_req, last_steps_per_req, duration_per_req, preemption_occurrence_per_req)
            for a, f, l, d, p in zipped:
                write_row(csv, [a, f, l, d, p])


def main(args: argparse.Namespace) -> None:
    x, ys = parse_data(args.path, args.num_requests, args.num_steps)
    # x, ys = correction(x, ys) # no correction needed in vllm

    if not args.skip_analysis:
        path = args.path.split("/")[-1].removesuffix(".csv")
        path += f"_n_{args.num_requests}" if args.num_requests != -1 else ""
        path += f"_s_{args.num_steps}" if args.num_steps != -1 else ""
        path += "_analyzed.csv"
        analyze(x, ys, path)

    if not args.skip_visualization:
        # colors = matplotlib.colors.XKCD_COLORS
        matplotlib.pyplot.figure(figsize=(((len(x) + 20) / 100), 8))
        matplotlib.pyplot.stackplot(x, *ys, colors=COLORS)
        matplotlib.pyplot.xlim(xmin=0)
        matplotlib.pyplot.ylim(ymin=0, ymax=33)
        matplotlib.pyplot.savefig(args.path.removesuffix(".csv") + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, help="The path of the csv file")
    parser.add_argument("-n", "--num-requests", type=int, default=-1, help="The number of requests to parse")
    parser.add_argument("-s", "--num-steps", type=int, default=-1, help="The number of steps to parse")
    parser.add_argument("--skip-analysis", action="store_true", default=False, help="Whether to skip analysis")
    parser.add_argument("--skip-visualization", action="store_true", default=False, help="Whether to skip visualization")

    args = parser.parse_args()
    main(args)
