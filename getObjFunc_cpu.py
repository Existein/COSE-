import os
import subprocess
import requests
import time
import csv

PROJECT_ID = "woven-ceremony-418503"
REGION = "us-central1"
FUNCTION_NAME = "cpu-intensive"

BASE_URL = f"https://us-central1-woven-ceremony-418503.cloudfunctions.net/cpu-intensive"
CONFIGURATIONS = [
    (128, 0.083), (128, 0.167), (128, 0.333), (128, 0.583), (128, 1), (128, 2),
    (256, 0.083), (256, 0.167), (256, 0.333), (256, 0.583), (256, 1), (256, 2),
    (512, 0.083), (512, 0.167), (512, 0.333), (512, 0.583), (512, 1), (512, 2),
    (1024, 0.583), (1024, 1), (1024, 2), 
    (2048, 1), (2048, 2), (2048, 4),
    (4096, 1), (4096, 2), (4096, 4), (4096, 6), (4096, 8), 
    (8192, 2), (8192, 4), (8192, 6), (8192, 8), (16384, 4), (16384, 6), (16384, 8)
]

def benchmark_configuration(memory_mi, vcpus):
    results = []

    for _ in range(3):  # 3 warm-up runs
        requests.get(BASE_URL)  # Discard results
    
    time.sleep(5)

    for _ in range(10):  # 10 main runs
        start_time = time.time()
        response = requests.get(BASE_URL)  # Make HTTP request
        end_time = time.time()
        results.append(end_time - start_time)
    return sum(results) / len(results)  # Calculate average execution time

def update_function_config(memory_mi, vcpus):
    # Construct the serverless function
    deploy_command = f"gcloud functions deploy cpu-intensive \
        --source=https://source.developers.google.com/projects/woven-ceremony-418503/repos/cpu-intensive \
        --trigger-http \
        --runtime=python312 \
        --allow-unauthenticated \
        --entry-point=cpu_http \
        --region=us-central1 \
        --gen2 \
        --memory={memory_mi}Mi \
        --cpu={vcpus}"

    # Execute the command
    output = subprocess.run(deploy_command, shell=True, check=True, capture_output=True)
    if output.stdout:
        print(f"Update complete: Memory (MiB): {memory_mi}, vCPUs: {vcpus}")

    # A short wait might be beneficial to let deployment complete
    time.sleep(5)  

if __name__ == "__main__":
    with open('cpu_benchmark_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Memory (MiB)', 'vCPUs', 'Avg. Execution Time'])  # Header row

        # random.shuffle(CONFIGURATIONS)  # Randomize order
        for mem, vcpu in CONFIGURATIONS:
            update_function_config(mem, vcpu)
            avg_time = benchmark_configuration(mem, vcpu)
            writer.writerow([mem, vcpu, avg_time])  # Write results for each configuration
            writer.writerow('')
            print(f"Memory: {mem}MiB, vCPUs: {vcpu}, Avg. Time: {avg_time}")
