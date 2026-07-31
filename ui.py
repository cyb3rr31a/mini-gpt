import gradio as gr # type:ignore
import requests # type:ignore

def generate_response(message, history):
    """
    Gradio automatically passes the new message and the chat history into this function.
    In 'messages' mode, message is a string, and history is a list of dictionaries.
    """
    try:
        response = requests.post(
            "http://localhost:8000/generate",
            json={"prompt": message, "max_new_tokens": 50},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()

            if "error" in result:
                return f"Backend Error: {result['error']}"
            generated_text = result.get("generated_tokens", "")

            if not generated_text or generated_text.strip() == "":
                return "⚠️ API returned an empty string."
            
            return str(generated_text)
            
        else:
            return f"HTTP Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return "🚨 Connection Refused! Make sure your FastAPI server is running."

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=generate_response,
    title="🎭 Shakespeare-GPT",
    description="Introducing GPT for Modern English enthusiasts"
)

if __name__ == "__main__":
    demo.launch()