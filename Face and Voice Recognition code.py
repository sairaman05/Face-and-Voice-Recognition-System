import cv2
import mysql.connector
import numpy as np
import os
import speech_recognition as sr
import time  

# MySQL connection details
DB_CONFIG = {
    'user': 'root',
    'password': 'sairaman2005',
    'host': 'localhost',
    'database': 'auth_system'
}       

# Initialize MySQL connection
def connect_database():
    return mysql.connector.connect(**DB_CONFIG)

# Store face data and voice password in MySQL
def store_user(name, face_model_path, voice_password):
    conn = connect_database()
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO users (name, face_data, voice_password) VALUES (%s, %s, %s)
    ''', (name, face_model_path, voice_password))

    conn.commit()
    cursor.close()
    conn.close()

# Retrieve all users from MySQL
def get_all_users():
    conn = connect_database()
    cursor = conn.cursor()

    cursor.execute('SELECT name, face_data, voice_password FROM users')
    rows = cursor.fetchall()

    users = []
    for row in rows:
        name = row[0]
        face_model_path = row[1].decode('utf-8')  # Decode from binary to string
        voice_password = row[2]
        users.append((name, face_model_path, voice_password))

    cursor.close()
    conn.close()
    return users

# Register face data using OpenCV's LBPH face recognizer
def register_face(name):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_samples = []
    ids = []
    sample_count = 0
    id = 0  # We will store only one user at a time
    
    print("Look into the camera and wait for your face to be captured.")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            sample_count += 1
            face_samples.append(gray[y:y + h, x:x + w])
            ids.append(id)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('Registering Face', frame)
        
        if sample_count >= 10:  # Collect 10 face samples
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(face_samples) > 0:
        face_recognizer.train(face_samples, np.array(ids))
        model_path = f'{name}_face_model.xml'
        face_recognizer.save(model_path)  # Save the model as an XML file
        print("Face registration successful!")
        return model_path  # Return the model path
    else:
        print("Face registration failed!")
        return None

# Register voice password using speech recognition
def register_voice_password():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        print("Please say your voice password:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        voice_password = recognizer.recognize_google(audio)
        print(f"Voice password captured: {voice_password}")
        return voice_password
    except sr.UnknownValueError:
        print("Could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

# Face recognition using OpenCV's LBPH face recognizer
def recognize_face(face_model_path, max_attempts=20, wait_time=3):
    # Ensure face_model_path is a valid string
    if not isinstance(face_model_path, str):
        print(f"Error: face_model_path must be a string, got {type(face_model_path)} instead.")
        return False
    
    if not os.path.exists(face_model_path):
        print(f"Error: The file '{face_model_path}' does not exist!")
        return False
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Load pre-trained face model from the given path
    face_recognizer.read(face_model_path)
    
    cap = cv2.VideoCapture(0)
    attempt_count = 0  # Track the number of attempts
    
    while attempt_count < max_attempts:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face_id, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])
            print(f"Face ID: {face_id}, Confidence: {confidence}")
            
            # Adjust confidence threshold: Lower confidence value indicates a better match.
            if confidence < 60:  # Modify this threshold based on your testing
                print(f"Face recognized with confidence {confidence}")
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                print(f"Unrecognized face (Confidence: {confidence})")
        
        attempt_count += 1
        print(f"Attempt {attempt_count} of {max_attempts}")
        
        # Show the video frame
        cv2.imshow('Face Recognition', frame)
        
        # Wait for 3 seconds before the next attempt
        time.sleep(wait_time)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("Face not recognized after maximum attempts.")
    cap.release()
    cv2.destroyAllWindows()
    return False

# Voice recognition using speech recognition
def recognize_voice(voice_password):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Please say your voice password:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        recognized_text = recognizer.recognize_google(audio)
        print(f"You said: {recognized_text}")
        return recognized_text == voice_password
    except sr.UnknownValueError:
        print("Could not understand the audio")
        return False
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return False

# Two-step authentication
def two_step_auth():
    users = get_all_users()

    if len(users) == 0:
        print("No users found in the system.")
        return

    name = input("Enter your name: ")
    matching_user = next((user for user in users if user[0] == name), None)

    if matching_user:
        face_model_path = matching_user[1]
        voice_password = matching_user[2]

        if face_model_path and isinstance(face_model_path, str):
            print("Look into the camera for face recognition.")
            if recognize_face(face_model_path):
                print("Face recognized successfully.")
                
                if recognize_voice(voice_password):
                    print("Voice password matched! \n\n Access granted.")
                else:
                    print("Voice password mismatch! \n\nAccess denied.")
            else:
                print("Face not recognized!\n\n Access denied.")
        else:
            print(f"Error: face_model_path is not a valid string or is empty. Received: {face_model_path}")
    else:
        print(f"User {name} not found.")

# Register a new user
def register_new_user():
    name = input("Enter your name: ")
    face_model_path = register_face(name)
    
    if face_model_path:
        voice_password = register_voice_password()
        if voice_password:
            store_user(name, face_model_path, voice_password)

# Main control flow
if __name__ == "__main__":
    while True:
        print("\n1. Register a new user")
        print("2. Perform two-step authentication")
        choice = int(input("Enter choice: "))

        if choice == 1:
            register_new_user()
        elif choice == 2:
            two_step_auth()
        else:
            print("Invalid choice")
