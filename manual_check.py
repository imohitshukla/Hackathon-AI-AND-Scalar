import requests

HF_URL = "https://iiimohitshukla-warehouse-robot-env.hf.space"

def check_endpoints():
    print(f"Checking {HF_URL}...")
    
    # 1. Check health
    r = requests.get(f"{HF_URL}/health")
    print(f"\n[GET /health] -> {r.json()}")
    
    # 2. Check tasks
    r = requests.get(f"{HF_URL}/tasks")
    tasks = r.json().get("tasks", [])
    print(f"\n[GET /tasks] -> {tasks}")

    # 3. Step through a task manually
    for t in tasks:
        print(f"\n--- Testing Task: {t} ---")
        
        # Reset
        res = requests.post(f"{HF_URL}/reset", json={"task_name": t}).json()
        print(f"[Reset] score: {res.get('score')} | reward: {res.get('reward')}")
        
        # Take a few steps to see reward changes
        for action in ["move_down", "move_right", "move_down", "move_left", "pickup"]:
            step_res = requests.post(f"{HF_URL}/step", json={"action": action}).json()
            score = step_res.get('score')
            reward = step_res.get('reward')
            print(f"[Step '{action}'] score: {score} | reward: {reward}")
            
            # Warn if exactly 0 or 1
            if score == 0.0 or score == 1.0 or reward == 0.0 or reward == 1.0:
                print(f"❌ WARNING: score or reward hit exactly 0.0 or 1.0!")
        
        # Check Final State
        state_res = requests.get(f"{HF_URL}/state").json()
        print(f"[State] score: {state_res.get('score')} | reward: {state_res.get('reward')}")

if __name__ == "__main__":
    check_endpoints()
