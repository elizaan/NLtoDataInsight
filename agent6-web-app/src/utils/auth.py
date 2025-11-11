from flask import request, jsonify
from functools import wraps

# Mock function to simulate user authentication
def is_authenticated():
    # Here you would implement your actual authentication logic
    return True

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            return jsonify({"message": "Unauthorized access"}), 401
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    # This function would retrieve the current user's information
    return {"username": "example_user"}  # Mock user data

def check_permissions(permission):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = get_current_user()
            # Here you would check if the user has the required permissions
            if permission not in user.get("permissions", []):
                return jsonify({"message": "Forbidden"}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator