import io
import os
import time
import traceback
from dataclasses import dataclass, field

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import spaces
import xxhash
from datasets import Audio

# Import configuration
import config

# Initialize API client based on configuration
api_provider = config.API_PROVIDER
if api_provider == 'groq':
    import groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Please set the GROQ_API_KEY environment variable.")
    client = groq.Client(api_key=api_key)
elif api_provider == 'openai':
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    client = openai.OpenAI(api_key=api_key)
else:
    raise ValueError(f"Unsupported API provider: {api_provider}")

def process_whisper_response(completion, provider):
    """
    Process Whisper transcription response and return text or null based on no_speech_prob
    
    Args:
        completion: Whisper transcription response object
        provider: The API provider ('openai' or 'groq')
        
    Returns:
        str or None: Transcribed text if no_speech_prob <= 0.7, otherwise None
    """
    if provider == 'groq':
        if completion.segments and len(completion.segments) > 0:
            no_speech_prob = completion.segments[0].get('no_speech_prob', 0)
            print("No speech prob:", no_speech_prob)

            if no_speech_prob > 0.7:
                return None
                
            return completion.text.strip()
    else:  # OpenAI
        # OpenAI doesn't provide a no_speech_prob in the same way
        # We'll just return the text
        if hasattr(completion, 'text'):
            return completion.text.strip()
        else:
            return completion
    
    return None

def transcribe_audio(client, file_name, provider):
    if file_name is None:
        return None

    try:
        with open(file_name, "rb") as audio_file:
            if provider == 'groq':
                response = client.audio.transcriptions.with_raw_response.create(
                    model=config.MODELS['groq']['audio'],
                    file=("audio.wav", audio_file),
                    response_format="verbose_json",
                )
                completion = process_whisper_response(response.parse(), provider)
            else:  # OpenAI
                audio_file_obj = open(file_name, "rb")
                response = client.audio.transcriptions.create(
                    model=config.MODELS['openai']['audio'],
                    file=audio_file_obj,
                    response_format="text",
                    language="en"
                )
                audio_file_obj.close()
                completion = process_whisper_response(response, provider)
            
            print(completion)
            
        return completion
    except Exception as e:
        print(f"Error in transcription: {e}")
        return f"Error in transcription: {str(e)}"


# Function to generate AI chat response using configured model
def generate_chat_completion(client, history, provider):
    messages = []
    messages.append(
        {
            "role": "system",
            "content": config.SYSTEM_PROMPT,
        }
    )

    for message in history:
        messages.append(message)

    try:
        if provider == 'groq':
            # Use Llama model from Groq
            completion = client.chat.completions.create(
                model=config.MODELS['groq']['chat'],
                messages=messages,
            )
            assistant_message = completion.choices[0].message.content
        else:  # OpenAI
            # Use GPT-4o mini from OpenAI
            completion = client.chat.completions.create(
                model=config.MODELS['openai']['chat'],
                messages=messages,
            )
            assistant_message = completion.choices[0].message.content
            
        return assistant_message
    except Exception as e:
        return f"Error in generating chat completion: {str(e)}"


@dataclass
class AppState:
    conversation: list = field(default_factory=list)
    stopped: bool = False
    model_outs: any = None


def process_audio(audio: tuple, state: AppState):
    return audio, state


@spaces.GPU(duration=40, progress=gr.Progress(track_tqdm=True))
def response(state: AppState, audio: tuple):
    if not audio:
        return AppState()

    file_name = f"/tmp/{xxhash.xxh32(bytes(audio[1])).hexdigest()}.wav"

    sf.write(file_name, audio[1], audio[0], format="wav")

    # Initialize client based on provider
    provider = config.API_PROVIDER
    if provider == 'groq':
        import groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Please set the GROQ_API_KEY environment variable.")
        client = groq.Client(api_key=api_key)
    else:  # OpenAI
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        client = openai.OpenAI(api_key=api_key)

    # Transcribe the audio file
    transcription = transcribe_audio(client, file_name, provider)
    if transcription:
        if transcription.startswith("Error"):
            transcription = "Error in audio transcription."

        # Append the user's message in the proper format
        state.conversation.append({"role": "user", "content": transcription})

        # Generate assistant response
        assistant_message = generate_chat_completion(client, state.conversation, provider)

        # Append the assistant's message in the proper format
        state.conversation.append({"role": "assistant", "content": assistant_message})

        print(state.conversation)

        # Optionally, remove the temporary file
        os.remove(file_name)

    return state, state.conversation


def start_recording_user(state: AppState):
    return None


theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#82000019",
        c200="#82000033",
        c300="#8200004c",
        c400="#82000066",
        c50="#8200007f",
        c500="#8200007f",
        c600="#82000099",
        c700="#820000b2",
        c800="#820000cc",
        c900="#820000e5",
        c950="#820000f2",
    ),
    secondary_hue="rose",
    neutral_hue="stone",
)

js = """
async function main() {
  const script1 = document.createElement("script");
  script1.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js";
  document.head.appendChild(script1)
  const script2 = document.createElement("script");
  script2.onload = async () =>  {
    console.log("vad loaded") ;
    var record = document.querySelector('.record-button');
    record.textContent = "Just Start Talking!"
    record.style = "width: fit-content; padding-right: 0.5vw;"
    const myvad = await vad.MicVAD.new({
      onSpeechStart: () => {
        var record = document.querySelector('.record-button');
        var player = document.querySelector('#streaming-out')
        if (record != null && (player == null || player.paused)) {
          console.log(record);
          record.click();
        }
      },
      onSpeechEnd: (audio) => {
        var stop = document.querySelector('.stop-button');
        if (stop != null) {
          console.log(stop);
          stop.click();
        }
      }
    })
    myvad.start()
  }
  script2.src = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js";
  script1.onload = () =>  {
    console.log("onnx loaded") 
    document.head.appendChild(script2)
  };
}
"""

js_reset = """
() => {
  var record = document.querySelector('.record-button');
  record.textContent = "Just Start Talking!"
  record.style = "width: fit-content; padding-right: 0.5vw;"
}
"""

with gr.Blocks(theme=theme, js=js) as demo:
    with gr.Row():
        input_audio = gr.Audio(
            label="Input Audio",
            sources=["microphone"],
            type="numpy",
            streaming=False,
            waveform_options=gr.WaveformOptions(waveform_color="#B83A4B"),
        )
    with gr.Row():
        chatbot = gr.Chatbot(label="Conversation", type="messages")
    state = gr.State(value=AppState())
    stream = input_audio.start_recording(
        process_audio,
        [input_audio, state],
        [input_audio, state],
    )
    respond = input_audio.stop_recording(
        response, [state, input_audio], [state, chatbot]
    )
    restart = respond.then(start_recording_user, [state], [input_audio]).then(
        lambda state: state, state, state, js=js_reset
    )

    cancel = gr.Button("New Conversation", variant="stop")
    cancel.click(
        lambda: (AppState(), gr.Audio(recording=False)),
        None,
        [state, input_audio],
        cancels=[respond, restart],
    )

if __name__ == "__main__":
    demo.launch()
