"""
notifier.py - Email alert system for disease-vector detection
"""

import smtplib
from email.message import EmailMessage

def send_alert_email(image_path, species, confidence, location):
    msg = EmailMessage()
    msg["Subject"] = "Mosquito Alert: Disease Vector Detected"
    msg["From"] = "noreply@mosquitotrap.ai"
    msg["To"] = "publichealth@agency.org"

    msg.set_content(f"A {species} mosquito with {confidence*100:.2f}% confidence was detected at {location}. See attached image.")

    with open(image_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="mosquito.jpg")

    try:
        with smtplib.SMTP("smtp.example.com", 587) as server:
            server.starttls()
            server.login("username", "password")
            server.send_message(msg)
    except Exception as e:
        print("Failed to send email:", e)
