from marshmallow import Schema, fields

class AnimationRequestSchema(Schema):
    phenomenon_description = fields.String(required=True)
    region_params = fields.Dict(required=True)
    output_id = fields.String(required=False)

class AnimationResponseSchema(Schema):
    animation_url = fields.String(required=True)
    log_file = fields.String(required=True)
    message = fields.String(required=False)

class DatasetSchema(Schema):
    dataset_name = fields.String(required=True)
    dataset_file = fields.String(required=True)

class ErrorResponseSchema(Schema):
    error = fields.String(required=True)
    message = fields.String(required=True)