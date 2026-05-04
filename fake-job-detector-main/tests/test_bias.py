
import torch
from predict import predict, extract_features, normalize_text, DigitalFootprintClassifier, tokenizer, DEVICE
import numpy as np

# Sample Real Job (Typical Job)
real_job = """
Software Engineer - Backend
We are looking for a Senior Backend Engineer to join our team at Google. 
You will be responsible for designing and implementing scalable services.
Requirements: 
- 5+ years of experience in Java or Python.
- Strong knowledge of distributed systems.
- Experience with cloud platforms (AWS/GCP).
Location: Bangalore, India.
Apply on our careers page or via LinkedIn.
"""

# Sample Fake Job (Typical Scam)
fake_job = """
Urgent Requirement: Data Entry Operator
Work From Home - Earn 2000 to 5000 Daily
No Interview Required - Direct Joining
Qualification: 10th/12th pass
Age: 18 to 45 years
Contact HR Priya: 9876543210 (WhatsApp only)
Registration fee: Rs. 500 (Refundable)
Send your resume to quickjobs@gmail.com immediately.
"""

def diagnostic():
    print("\n--- DIAGNOSTIC START ---")
    
    for name, text in [("REAL JOB", real_job), ("FAKE JOB", fake_job)]:
        print(f"\n>>> TESTING: {name}")
        norm = normalize_text(text)
        features = extract_features(norm)
        print("Features:", features)
        
        prob = predict(norm)
        score = round(prob * 100)
        print(f"Prediction Probability: {prob:.4f}")
        print(f"Final Score: {score}%")
        
        if score > 65:
            print("Status: FAKE")
        elif score < 35:
            print("Status: REAL")
        else:
            print("Status: SUSPICIOUS")

if __name__ == "__main__":
    diagnostic()
