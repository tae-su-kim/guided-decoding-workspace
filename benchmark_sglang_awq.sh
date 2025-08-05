#! /bin/bash

MODEL_NAME="/workspace/Qwen3-32B-AWQ"
TOKENIZER="/workspace/Qwen3-32B-AWQ"
host="127.0.0.1"
port=30000
TP=4
MAX_RUNNING_REQUESTS=32
ENABLING_REASONING=1

source /workspace/env/bin/activate

# Function to check if server is ready
check_server_health() {
    local max_attempts=180  # 30 minutes timeout
    local attempt=1
    
    echo "Checking server health..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://${host}:${port}/v1/models > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: Server not ready yet, waiting..."
        sleep 10
        ((attempt++))
    done
    
    echo "Server failed to start within timeout"
    return 1
}

# Function to cleanup server
cleanup_server() {
    echo "Shutting down server..."
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 
    fi
    # Also kill any remaining vllm processes
    pkill -f "sglang.launch_server"

    echo "Waiting for server to shutdown..."
    if [ ! -z "$SERVER_PID" ]; then
        wait $SERVER_PID
    fi
    sleep 10
    echo "Server shutdown completed"
}

# Set trap to cleanup on script exit
trap cleanup_server EXIT

for REASONING_BUDGET in 300; do
    benchmark_dir="Qwen3_32B_SGLang/xgrammar-guided-reasoning-${REASONING_BUDGET}-budget-awq"
    mkdir -p ${benchmark_dir}

    for VUSER in 32 16 8; do
        echo "Starting SGLang server with xgrammar and reasoning..."
        
        # Start SGLang server in background
        REASONING_BUDGET=${REASONING_BUDGET} \
        python3 -m sglang.launch_server \
            --model-path ${MODEL_NAME} \
            --tokenizer-path ${TOKENIZER} \
            --host ${host} \
            --port ${port} \
            --tool-call-parser qwen25 \
            --tp 4 \
            --reasoning-parser qwen3 \
            --grammar-backend xgrammar \
            --max-running-requests ${MAX_RUNNING_REQUESTS} \
            --mem-fraction-static 0.9 \
            --context-length 16384 \
            --enable-mixed-chunk \
            --enable-tokenizer-batch-encode \
            --cuda-graph-bs 1 2 3 4 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
            >& ${benchmark_dir}/sglang_server_${VUSER}.log &


        SERVER_PID=$!
        echo "Server started with PID: $SERVER_PID"
        
        # Wait for server to be ready
        if ! check_server_health; then
            echo "Failed to start server, skipping this configuration"
            cleanup_server
            continue
        fi

        # xgrammar warmup
        python3 ./benchmark/benchmark_serving_structured_output.py \
            --endpoint /v1/chat/completions \
            --backend openai-chat \
            --target-engine-type sglang \
            --host ${host} \
            --port ${port} \
            --dataset xgrammar_bench \
            --model ${MODEL_NAME}  \
            --tokenizer ${TOKENIZER} \
            --output-len 1 \
            --save-results \
            --result-filename ${benchmark_dir}/dummyrun_vuser_${VUSER}.json \
            --num-prompts 100 \
            --percentile-metrics ttft,tpot,itl,e2el \
            --metric-percentiles 60,70,80,90,95,99 \
            --max-concurrency $VUSER \
            --guided 1 \
            --reasoning 1 \
            --dummy-content \
            2>&1 | tee ${benchmark_dir}/dummyrun_vuser_${VUSER}.log 

        python3 ./benchmark/benchmark_serving_structured_output.py \
            --endpoint /v1/chat/completions \
            --backend openai-chat \
            --target-engine-type sglang \
            --host ${host} \
            --port ${port} \
            --dataset xgrammar_bench \
            --model ${MODEL_NAME}  \
            --tokenizer ${TOKENIZER} \
            --output-len 640 \
            --save-results \
            --result-filename ${benchmark_dir}/vuser_${VUSER}.json \
            --num-prompts 300 \
            --percentile-metrics ttft,tpot,itl,e2el \
            --metric-percentiles 60,70,80,90,95,99 \
            --max-concurrency $VUSER \
            --guided 1 \
            --reasoning 1 \
            2>&1 | tee ${benchmark_dir}/benchmark_vuser_${VUSER}.log 

        cleanup_server
        SERVER_PID=""
        echo "Waiting a bit before next configuration..."
        sleep 10

    done
    
    python3 json_to_csv.py --json_dir ${benchmark_dir} > ${benchmark_dir}/summary.txt
    cat ${benchmark_dir}/summary.txt
    # Cleanup server before next iteration

done
