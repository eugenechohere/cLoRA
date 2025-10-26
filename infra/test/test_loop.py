import requests
import time

def train_and_update(data_path):
    url = "http://localhost:8001/train-and-update"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "data_path": data_path
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status() 
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

def main():
    data_path = '/home/ubuntu/calhacks-continual-learning/infra/data/batch_256_hehe.jsonl'
    num_iterations = 1  # Number of times to run the loop
    delay_seconds = 1  # Delay between iterations (in seconds)
    
    print(f"Starting training loop with {num_iterations} iterations...")
    print(f"Delay between iterations: {delay_seconds} seconds\n")
    
    for i in range(num_iterations):
        print(f"Iteration {i + 1}/{num_iterations}")
        print(f"Sending request with data_path: {data_path}")
        
        response = train_and_update(data_path)
        
        if response:
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
        else:
            print("Request failed")
        
        # Don't sleep after the last iteration
        if i < num_iterations - 1:
            print(f"Waiting {delay_seconds} seconds before next iteration...\n")
            time.sleep(delay_seconds)
        else:
            print("\nAll iterations completed!")

if __name__ == "__main__":
    main()