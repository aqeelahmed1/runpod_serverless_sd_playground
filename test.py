
import base64
from PIL import Image
from io import BytesIO
import asyncio
import aiohttp
import os
import runpod
from runpod import AsyncioEndpoint, AsyncioJob


# add your runpod api key and endpoint url
runpod.api_key = ""
end_point=""


# Assuming your base64 encoded string is stored in a variable called 'base64_image'
def decode(base64_image):
    image_bytes = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save('output_image.jpg')  # You can change the format and file name as needed


async def main():
    async with aiohttp.ClientSession() as session:
        input_payload = {"prompt": "astronaut riding a horse"}
        endpoint = AsyncioEndpoint(end_point, session)
        job: AsyncioJob = await endpoint.run(input_payload)

        # Polling job status
        while True:
            status = await job.status()
            print(f"Current job status: {status}")
            if status == "COMPLETED":
                output = await job.output()
                print("Job output:", output)
                decode(output)
                break  # Exit the loop once the job is completed.
            elif status in ["FAILED"]:
                print("Job failed or encountered an error.")

                break
            else:
                print("Job in queue or processing. Waiting 3 seconds...")
                await asyncio.sleep(3)  # Wait for 3 seconds before polling again


if __name__ == "__main__":
    asyncio.run(main())
