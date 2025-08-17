import dspy
import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig

# Initialize Vertex AI — adjust region/project accordingly
vertexai.init(project="your-gcp-project-id", location="us-central1")

class GeminiVertex(dspy.SignatureLM):
    def __init__(self, model_name="gemini-1.5-pro", max_tokens=2048, temperature=0.7, **kwargs):
        super().__init__()
        self.model = GenerativeModel(model_name)
        self.generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def __call__(self, prompt, **kwargs):
        # Generate response using Vertex AI Gemini
        response = self.model.generate_content(
            prompt,
            generation_config=self.generation_config
        )
        return response.text

# Configure DSPy to use this Gemini wrapper
dspy.settings.configure(lm=GeminiVertex())


class SimpleQA(dspy.Signature):
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

qa = dspy.Predict(SimpleQA)

response = qa(context="Mumbai is the financial capital of India.", question="What is the financial capital of India?")
print(response)
