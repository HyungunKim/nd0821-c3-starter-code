"""
Script to test the live API using the requests module.
"""
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API endpoint URL
# Replace with your actual Heroku app URL when deployed
API_URL = "http://localhost:8000"

def test_api_get():
    """
    Test the GET endpoint of the API.
    """
    response = requests.get(f"{API_URL}/")
    logger.info(f"GET Response Status Code: {response.status_code}")
    logger.info(f"GET Response Body: {response.json()}")
    print(response.json())
    return response.status_code == 200

def test_api_post_high_income():
    """
    Test the POST endpoint of the API with sample data for high income prediction (>50K).
    """
    # Sample data for someone likely to earn > 50K
    data = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 160323,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 10000,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }
    
    # Make a prediction
    response = requests.post(f"{API_URL}/predict", json=data)
    logger.info(f"POST Response Status Code: {response.status_code}")
    
    if response.status_code == 200:
        logger.info(f"POST Response Body: {response.json()}")
        logger.info(f"Prediction: {response.json()['prediction']}")
        logger.info(f"Probability: {response.json()['probability']}")
    else:
        logger.error(f"POST Request Failed: {response.text}")
    print(response.json())
    return response.status_code == 200

def test_api_post_low_income():
    """
    Test the POST endpoint of the API with sample data for low income prediction (<=50K).
    """
    # Sample data for someone likely to earn <= 50K
    data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Never-married",
        "occupation": "Service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }
    
    # Make a prediction
    response = requests.post(f"{API_URL}/predict", json=data)
    logger.info(f"POST Response Status Code: {response.status_code}")
    
    if response.status_code == 200:
        logger.info(f"POST Response Body: {response.json()}")
        logger.info(f"Prediction: {response.json()['prediction']}")
        logger.info(f"Probability: {response.json()['probability']}")
    else:
        logger.error(f"POST Request Failed: {response.text}")
    print(response.json())
    return response.status_code == 200

def main():
    """
    Main function to test the API.
    """
    logger.info("Testing the Census Income Prediction API")
    
    # Test GET endpoint
    get_success = test_api_get()
    if get_success:
        logger.info("GET endpoint test passed")
    else:
        logger.error("GET endpoint test failed")
    
    # Test POST endpoint for both income cases
    post_success_high = test_api_post_high_income()
    post_success_low = test_api_post_low_income()
    
    if post_success_high and post_success_low:
        logger.info("POST endpoint tests passed")
    else:
        logger.error("POST endpoint tests failed")
    
    # Overall result
    if get_success and post_success_high and post_success_low:
        logger.info("All API tests passed")
    else:
        logger.error("Some API tests failed")

if __name__ == "__main__":
    main()