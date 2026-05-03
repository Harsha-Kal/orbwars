import sys
import os
import json
import kaggle
import requests

def download_replay(episode_id):
    api = kaggle.KaggleApi()
    api.authenticate()
    
    # Kaggle API authentication uses Basic Auth with (username, key)
    # from your kaggle.json file.
    username = api.config_values.get('username')
    key = api.config_values.get('key')
    
    # Note: The correct URL endpoint often uses 'episode' (singular)
    url = f"https://www.kaggle.com/api/v1/competitions/episode/replay/{episode_id}"
    
    print(f"Downloading replay for episode {episode_id}...")
    
    try:
        response = requests.get(url, auth=(username, key))
        
        # If 'episode' didn't work, try 'episodes'
        if response.status_code == 404:
            url = f"https://www.kaggle.com/api/v1/competitions/episodes/replay/{episode_id}"
            response = requests.get(url, auth=(username, key))
            
        response.raise_for_status()
        
        data = response.json()
        filename = f"replay_{episode_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f)
            
        print(f"Successfully saved replay to: {os.path.abspath(filename)}")
        
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Tip: If you get a 403, check that you are joined to the competition on Kaggle.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_replay.py <episode_id>")
    else:
        download_replay(sys.argv[1])
