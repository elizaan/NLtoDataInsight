def generate_animation_params(phenomenon, region_params):
    # Function to generate parameters for animation based on selected phenomenon and region
    params = {
        "phenomenon": phenomenon,
        "region": region_params,
        "resolution": "high",  # Default resolution
        "format": "gif"        # Default output format
    }
    return params

def validate_input_data(data):
    # Function to validate input data for animation generation
    if not data.get("phenomenon"):
        raise ValueError("Phenomenon must be specified.")
    if not data.get("region"):
        raise ValueError("Region parameters must be provided.")
    # Additional validation logic can be added here
    return True

def format_response(data):
    # Function to format the response data for API output
    return {
        "status": "success",
        "data": data
    }