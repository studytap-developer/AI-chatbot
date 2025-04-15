import firebase_admin
from firebase_admin import credentials, auth

# Step 1: Load your service account key
cred = credentials.Certificate("//Users//maheshkonijeti//Desktop//project//sign-in-5fb04-firebase-adminsdk-fbsvc-dcb4fdbf86.json")
firebase_admin.initialize_app(cred)

# Step 2: Try listing the first user (if any)
user_iterator = auth.list_users()
for user in user_iterator.users:
    print("✅ Connected! Found user:", user.uid)
    break  # Just print the first user and stop
else:
    print("✅ Connected to Firebase, but no users found.")
app = firebase_admin.get_app()
print("✅ Firebase initialized with name:", app.name)
