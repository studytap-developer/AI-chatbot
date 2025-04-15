import firebase_admin
from firebase_admin import credentials, auth

cred = credentials.Certificate("//Users//maheshkonijeti//Desktop//project//sign-in-5fb04-firebase-adminsdk-fbsvc-dcb4fdbf86.json")
firebase_admin.initialize_app(cred)
