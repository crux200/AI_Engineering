import gradio as gr
import spaces
@spaces.GPU
def respond(message,history):
    response=f"You said: {message} And I say I love learning AI engineering with SuperDataScience!"
    return response
gr.ChatInterface(fn=respond).launch()
