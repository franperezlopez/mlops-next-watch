import os

from dotenv import load_dotenv

from conf import globals, paths

# Load environment variables from .env file

load_dotenv(
    paths.get_path(
        paths.ENV,
        storage=globals.Storage.HOST,
    )
)

# Access the AWS credentials using the environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# Create the AWS credentials file with the [default] profile
aws_path = paths.get_path(".aws/credentials", storage=globals.Storage.HOST)
aws_path.parent.mkdir(parents=True, exist_ok=True)

with open(aws_path, "w") as f:
    f.write("[default]\n")
    f.write(f"aws_access_key_id = {aws_access_key_id}\n")
    f.write(f"aws_secret_access_key = {aws_secret_access_key}\n")

# Set the appropriate permissions on the file
os.chmod(aws_path, 0o600)
