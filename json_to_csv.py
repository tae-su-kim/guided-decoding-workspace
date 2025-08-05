import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--json_dir", type=str, required=True)
args = parser.parse_args()

# 헤더 출력
fields = [
    "correct_rate(%)",
    "vuser",
    'total_input_tokens',
    'total_output_tokens',
    'total_reasoning_tokens',
    'duration',
    'request_throughput',
    'output_throughput',
    'total_token_throughput',
    'mean_ttft_ms',
    'median_ttft_ms',
    'p60_ttft_ms',
    'p70_ttft_ms',
    'p80_ttft_ms',
    'p90_ttft_ms',
    'p95_ttft_ms',
    'p99_ttft_ms',
    'mean_tpot_ms',
    'median_tpot_ms',
    'p60_tpot_ms',
    'p70_tpot_ms',
    'p80_tpot_ms',
    'p90_tpot_ms',
    'p95_tpot_ms',
    'p99_tpot_ms',
    'mean_itl_ms',
    'median_itl_ms',
    'p60_itl_ms',
    'p70_itl_ms',
    'p80_itl_ms',
    'p90_itl_ms',
    'p95_itl_ms',
    'p99_itl_ms',
    'mean_e2el_ms',
    'median_e2el_ms',
    'p60_e2el_ms',
    'p70_e2el_ms',
    'p80_e2el_ms',
    'p90_e2el_ms',
    'p95_e2el_ms',
    'p99_e2el_ms'
]
print(",".join(fields))
# 헤더 출력
for vusers in (1, 4, 8, 16, 32):
    try:
        json_file = f'{args.json_dir}/vuser_{vusers}.json'
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        continue

    # 데이터 행 출력
    values = [
        str(data['correct_rate(%)']),
        str(vusers),
        str(data['total_input_tokens']),
        str(data['total_output_tokens']),
        str(data['total_reasoning_tokens']),
        str(data['duration']),
        str(data['request_throughput']),
        str(data['output_throughput']),
        str(data['total_token_throughput']),
        str(data['mean_ttft_ms']),
        str(data['median_ttft_ms']),
        str(data['p60_ttft_ms']),
        str(data['p70_ttft_ms']),
        str(data['p80_ttft_ms']),
        str(data['p90_ttft_ms']),
        str(data['p95_ttft_ms']),
        str(data['p99_ttft_ms']),
        str(data['mean_tpot_ms']),
        str(data['median_tpot_ms']),
        str(data['p60_tpot_ms']),
        str(data['p70_tpot_ms']),
        str(data['p80_tpot_ms']),
        str(data['p90_tpot_ms']),
        str(data['p95_tpot_ms']),
        str(data['p99_tpot_ms']),
        str(data['mean_itl_ms']),
        str(data['median_itl_ms']),
        str(data['p60_itl_ms']),
        str(data['p70_itl_ms']),
        str(data['p80_itl_ms']),
        str(data['p90_itl_ms']),
        str(data['p95_itl_ms']),
        str(data['p99_itl_ms']),
        str(data['mean_e2el_ms']),
        str(data['median_e2el_ms']),
        str(data['p60_e2el_ms']),
        str(data['p70_e2el_ms']),
        str(data['p80_e2el_ms']),
        str(data['p90_e2el_ms']),
        str(data['p95_e2el_ms']),
        str(data['p99_e2el_ms'])
    ]
    print(','.join(values))
