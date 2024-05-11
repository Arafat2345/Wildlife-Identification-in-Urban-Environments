from fastai.vision.all import *
import gradio as gr

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

Wildlife_labels = (
"urban armadillo",
"urban bat",
"urban bird",
"urban coyote",
"urban deer",
"urban fox ",
"urban hedgehog",
"urban opossum",
"urban pigeon",
"urban rabbit",
"urban raccoon",
"urban rat",
"urban skunk",
"urban squirrel",
"urban stray Cat"
)

model = load_learner('wildlife-recognizer-v2.pkl')

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(Wildlife_labels, map(float, probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label(num_top_classes=5)
examples = [
    'unknown_00.jpg',
    'unknown_01.jpg',
    'unknown_02.jpg',
    'unknown_03.jpg',
    'unknown_04.jpg',
    'unknown_05.jpg'
    ]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False)