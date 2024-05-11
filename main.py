from openai import OpenAI
import requests, os, time
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI"))

BASE_URL = "https://client.camb.ai/apis"
API_KEY = os.getenv("CAMBAI")
HEADERS = {"headers": {"x-api-key": API_KEY}}


prescription = open("input/prescription.txt", "r")
prescription_text = prescription.read()
prescription.close()


def transcribe_doctor_directions():
    audio_file= open("input/doctor_directions.mp3", "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    return transcription.text

def compile_doctor_directions(transcription, prescription):
    prompt = f"""
    Doctor's directions: 
    {transcription}

    Prescription details:
    {prescription}
    """

    response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": """
         You will help compile the doctor's directions and the prescription details to give out clear concise instructions for the patient in a casual day-to-day Arabic language.
         By compiling, I mean you should extract the important information from the doctor's directions and the prescription details and combine them as if you are having a natural convo with the patient. 
         Make sure the compiled instructions are in first person as if you are the doctor, ALWAYS WRITE IT AS IF IT IS A CONVERSATION."""},
        {"role": "user", "content": prompt},
    ]
    )

    return response.choices[0].message.content



def convert_directions_to_audio(compiled_directions, doctor_age=39):
    tts_payload = {
        "text": compiled_directions,
        "voice_id": 8985,  # Example voice ID
        "language": 4,  # Hindi
        "age": doctor_age,
    }

    res = requests.post(f"{BASE_URL}/tts", json=tts_payload, **HEADERS)
    task_id = res.json()["task_id"]

    while True:
        res = requests.get(f"{BASE_URL}/tts/{task_id}", **HEADERS)
        status = res.json()["status"]
        print(f"Polling: {status}")

        time.sleep(1.5)
        if status == "SUCCESS":
            run_id = res.json()["run_id"]

            print(f"Run ID: {run_id}")
            res = requests.get(f"{BASE_URL}/tts_result/{run_id}", **HEADERS, stream=True)
            with open("output/tts_output.wav", "wb") as f:
                for chunk in res.iter_content(chunk_size=1024):
                    f.write(chunk)
            print("Done!")
            break

if __name__ == "__main__":
    print("Transcribing doctor's directions...")
    transcription = transcribe_doctor_directions()
    print("Compiling doctor's directions...")
    compiled_directions = compile_doctor_directions(transcription, prescription_text)
    
    # Save compiled directions to a file
    print("Saving compiled directions to file...")
    print(compiled_directions)
    with open("output/compiled_directions.txt", "w", encoding="UTF-8") as file:
        file.write(compiled_directions)
    
    # Convert compiled directions to audio
    print("Converting compiled directions to audio...")
    convert_directions_to_audio(compiled_directions)