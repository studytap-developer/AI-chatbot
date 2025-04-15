from fastapi import Request, HTTPException, Depends
from firebase_admin import auth
from firebase_admin._auth_utils import InvalidIdTokenError

def get_current_user_id(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    id_token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token["uid"]  # or decoded_token["email"]
    except InvalidIdTokenError:
        raise HTTPException(status_code=403, detail="Invalid Firebase ID token")
