# Face-and-Voice-Recognition-System

Overview: 
  This project implements a two-step authentication system using face recognition and voice recognition to enhance security. The system:
        Registers users by capturing their face and voice password.
        Authenticates users by verifying their face and matching their voice password.
  The project uses OpenCV for face recognition, SpeechRecognition for voice authentication, and MySQL for storing user credentials.


How it Works ?

1. User Registration:
      The system will capture the users face and the face model is stored as .xml file.
      As a second step it asks for voice password
      Both the face model as well the voice password is stored in the mysql database.

2. User Authentication:
      While checking, this system fetches the face model and the voice password from the database. First the user face is checked using LBPH's recognizer.
      If the face matches, it goes on with the voice password.
      If both face as well as the voice matches with those in database, the user is given access otherwise the user is not allowed.
