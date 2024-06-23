from flask import Flask, request, jsonify
from pyngrok import ngrok
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, this is your Flask app!"

load_dotenv()

ngrok_auth_token = os.getenv('NGROK_AUTH_TOKEN')

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)

@app.route('/few-shot-prompt', methods=['POST'])
def few_shot_prompt():
    data = request.get_json()

    # Retrieve examples and the input prompt from the request
    examples = data.get('examples', [])
    prompt = data.get('prompt', '')

    # Construct the few-shot prompt
    few_shot_prompt = ""
    for example in examples:
        few_shot_prompt += f"Q: {example['question']}\nA: {example['answer']}\n\n"
    few_shot_prompt += f"Q: {prompt}\nA:"

    # Make a request to the OpenAI API using the new interface
    try:
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or another suitable model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": few_shot_prompt}
            ],
            max_tokens=150
        )
        print(stream)
        response = stream.choices[0].message.content
        # for chunk in stream:
            # response += chunk.choices[0].delta.content or ""

        # Extract and return the generated response
        # generated_text = response['choices'][0]['message']['content'].strip()
        return jsonify({"response": response})

    except openai.APIStatusError as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    ngrok.set_auth_token(ngrok_auth_token)

    # Set up a tunnel to the Flask server port
    public_url = ngrok.connect(5000).public_url
    print("Public URL:", public_url)

    # Run the Flask server
    app.run(port=5000)