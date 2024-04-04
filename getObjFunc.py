import requests
import time
import random
from google.cloud import functions_v2

PROJECT_ID = "woven-ceremony-418503"
REGION = "us-central1"
FUNCTION_NAME = "io-intensive"

BASE_URL = f"https://us-central1-woven-ceremony-418503.cloudfunctions.net/io-intensive"
CONFIGURATIONS = [
    (128, 0.083), (128, 0.167), (128, 0.333), (128, 0.583), (128, 1), (128, 2),
    (256, 0.083), (256, 0.167), (256, 0.333), (256, 0.583), (256, 1), (256, 2),
    (512, 0.083), (512, 0.167), (512, 0.333), (512, 0.583), (512, 1), (512, 2),
    (1024, 0.583), (1024, 1), (1024, 2), 
    (2048, 1), (2048, 2), (2048, 4),
    (4096, 1), (4096, 2), (4096, 4), (4096, 6), (4096, 8), 
    (8192, 2), (8192, 4), (8192, 6), (8192, 8), (16384, 4), (16384, 6), (16384, 8),
    (32768, 8)
]

def benchmark_configuration(memory_mb, vcpus):
    results = []

    for _ in range(3):  # 3 warm-up runs
        requests.get(BASE_URL)  # Discard results

    for _ in range(10):  # 10 main runs
        start_time = time.time()
        response = requests.get(BASE_URL)  # Make HTTP request
        end_time = time.time()
        results.append(end_time - start_time)
    return sum(results) / len(results)  # Calculate average execution time

def update_function_config(memory_mb, vcpus):
    client = functions_v2.CloudFunctionsServiceClient()
    # ... (Construct the function update request, see SDK docs)
    operation = client.update_function(request={
        'function': {'name': "...",  # Specify function details
                     'available_memory_mb': memory_mb,
                     # Other fields for vCPU etc.
                     }
    })
    operation.result()  # Wait for the operation to complete

if __name__ == "__main__":
    random.shuffle(CONFIGURATIONS)  # Randomize order
    for mem, vcpu in CONFIGURATIONS:
        update_function_config(mem, vcpu)  
        avg_time = benchmark_configuration(mem, vcpu)
        print(f"Memory: {mem}MB, vCPUs: {vcpu}, Avg. Time: {avg_time}")
