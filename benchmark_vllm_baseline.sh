#! /bin/bash

MODEL_NAME="/workspace/HF-Qwen3-32B"
TOKENIZER="/workspace/HF-Qwen3-32B"
host="127.0.0.1"
port=30000
REASONING_BUDGET=300
TP=4
DP=1
MAX_BATCH_SIZE=32

source /workspace/vllm/bin/activate

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
    pkill -f "vllm serve"

    echo "Waiting for server to shutdown..."
    if [ ! -z "$SERVER_PID" ]; then
        wait $SERVER_PID
    fi
    sleep 10
    echo "Server shutdown completed"
}

# Set trap to cleanup on script exit
trap cleanup_server EXIT

for ENABLING_REASONING in 1; do
    benchmark_dir="Qwen3_32B_benchmark_result/vllm_v1_TP${TP}_DP${DP}/1k_xgrammar-guided-reasoning-${REASONING_BUDGET}-budget"
    mkdir -p ${benchmark_dir}

    for VUSER in 32 16 8; do
        echo ---------------------------------------------------------------------------------------
        echo "Starting vLLM server with xgrammar and reasoning..."

        # Start vLLM server in background
        REASONING_BUDGET=${REASONING_BUDGET} \
        vllm serve ${MODEL_NAME} \
            --dtype bfloat16 \
            --host ${host} \
            --port ${port} \
            --tokenizer ${TOKENIZER} \
            --guided-decoding-backend xgrammar \
            --tensor-parallel-size ${TP} \
            --data-parallel-size ${DP} \
            --max-num-seqs ${MAX_BATCH_SIZE} \
            --max-model-len 8192 \
            --tool-call-parser hermes \
            --disable-log-requests \
            --reasoning-parser qwen3 \
            >& ${benchmark_dir}/vllm_server_${VUSER}.log &
        
        SERVER_PID=$!
        echo "Server started with PID: $SERVER_PID"
        
        # Wait for server to be ready
        if ! check_server_health; then
            echo "Failed to start server, skipping this configuration"
            cleanup_server
            continue
        fi

        python3 ./benchmark/benchmark_serving_structured_output.py \
            --endpoint /v1/chat/completions \
            --backend openai-chat \
            --target-engine-type vllm \
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
