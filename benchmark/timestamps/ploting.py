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




def main(args: argparse.Namespace) -> None:
    datas = []
    for path in args.paths:
        x, ys = parse_data(path, args.num_requests, args.num_steps)
        datas.append((x, ys, path))
    # x, ys = correction(x, ys) # no correction needed in vllm

    if not args.skip_visualization:
        # colors = matplotlib.colors.XKCD_COLORS
        num_plots = len(datas)
        
        # Create subplots arranged vertically
        fig, axes = matplotlib.pyplot.subplots(num_plots, 1, figsize=(((len(datas[0][0]) + 20) / 100), 8 * num_plots), sharex=True)
        
        # Handle single plot case
        if num_plots == 1:
            axes = [axes]
        
        # Plot each dataset in its own subplot
        for i, (x, ys, path) in enumerate(datas):
            axes[i].stackplot(x, *ys, colors=COLORS)
            axes[i].set_xlim(xmin=0)
            axes[i].set_ylim(ymin=0, ymax=33)
            axes[i].set_title(f'Dataset: {path}')
            axes[i].grid(True, alpha=0.3)
        
        # Set common x-axis label only on the bottom plot
        axes[-1].set_xlabel('Time Steps')
        
        # Set y-axis label for all plots
        for ax in axes:
            ax.set_ylabel('Active Requests')
        
        # Adjust layout to prevent overlap
        matplotlib.pyplot.tight_layout()
        
        # Save with combined filename
        output_name = "_".join([path.removesuffix(".csv").split("/")[-1] for _, _, path in datas])
        matplotlib.pyplot.savefig(f"combined_{output_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("paths", type=str, nargs="+", help="The path of the csv file")
    parser.add_argument("-n", "--num-requests", type=int, default=-1, help="The number of requests to parse")
    parser.add_argument("-s", "--num-steps", type=int, default=-1, help="The number of steps to parse")
    parser.add_argument("--skip-visualization", action="store_true", default=False, help="Whether to skip visualization")

    args = parser.parse_args()
    main(args)
